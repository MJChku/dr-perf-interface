/* Opt-in native synchronization observations supporting declared wait checks.
 * API observations do not assert sleeping, ownership, or causality by timing.
 * Kept separate from instruction exclusion (including CUDA/GX). */
#include <sys/syscall.h>

static int opt_max_wait_records = 100000;
static bool opt_waits;
static bool opt_wait_delay_event;
static char opt_wait_delay_region[RNAME_MAX];
static int opt_wait_delay_ms;
static int64 wait_delay_injections;
static void *wait_lock;
static int wait_count, wait_hooks;
static int64 wait_order, wait_dropped;

typedef struct {
    const char *name, *kind;
    int object_arg, aux_arg;
} wait_api_t;
static const wait_api_t wait_apis[] = {
    {"perfmark_release", "declared_publish", 0, 1},
    {"perfmark_wait", "declared_waited", 0, 1},
    {"perfmark_wait_null", "declared_waited_null", -1, -1},
    {"perfmark_event_publish", "declared_publish", 0, 1},
    {"perfmark_event_waited", "declared_waited", 0, 1},
    {"perfmark_waited_null", "declared_waited_null", -1, -1},
    {"perfmark_runtime_wait_begin", "runtime_wait_begin", 0, 1},
    {"perfmark_runtime_wait_end", "runtime_wait_end", 0, 1},
    {"perfmark_async_scope", "runtime_scope", 0, 1},
    {"sem_init", "sem_init", 0, 2}, {"sem_destroy", "sem_destroy", 0, -1},
    {"sem_post", "sem_publish", 0, -1}, {"sem_wait", "completion", 0, -1},
    {"sem_timedwait", "completion", 0, -1},
    {"sem_clockwait", "completion", 0, -1}, {"sem_trywait", "sem_consume", 0, -1},
    {"pthread_mutex_init", "mutex_init", 0, -1},
    {"pthread_mutex_destroy", "mutex_destroy", 0, -1},
    {"pthread_mutex_lock", "lock", 0, -1},
    {"pthread_mutex_timedlock", "lock", 0, -1},
    {"pthread_mutex_unlock", "unlock", 0, -1},
    {"pthread_cond_wait", "condition", 0, 1},
    {"pthread_cond_timedwait", "condition", 0, 1},
    {"pthread_cond_signal", "signal", 0, -1},
    {"pthread_cond_broadcast", "broadcast", 0, -1},
    {"pthread_join", "join", 0, -1},
    {"recv", "receive", 0, -1}, {"recvfrom", "receive", 0, -1},
    {"recvmsg", "receive", 0, -1},
    {"cudaEventCreate", "event_create", 0, -1},
    {"cudaEventCreateWithFlags", "event_create", 0, 1},
    {"cudaEventDestroy", "event_destroy", 0, -1},
    {"cudaEventRecord", "event_record", 0, 1},
    {"cudaEventRecordWithFlags", "event_record", 0, 1},
    {"cudaEventSynchronize", "event_wait", 0, -1},
    {"cudaStreamSynchronize", "stream_wait", 0, -1},
    {"cudaStreamCreate", "stream_create", 0, -1},
    {"cudaStreamCreateWithFlags", "stream_create", 0, -1},
    {"cudaStreamCreateWithPriority", "stream_create", 0, -1},
    {"cudaStreamDestroy", "stream_destroy", 0, -1},
    {"cudaDeviceSynchronize", "device_wait", -1, -1},
    {"cudaStreamWaitEvent", "stream_dependency", 0, 1},
    {"cudaMemcpyAsync", "transfer", 4, -1},
    {"cudaMemcpy2DAsync", "transfer", 7, -1},
};
static int64 wait_unattributed[sizeof(wait_apis) / sizeof(wait_apis[0])];
typedef struct {
    const wait_api_t *api;
    app_pc address;
    bool excluded;
} wait_hook_t;
static wait_hook_t wait_hook_info[512];
typedef struct {
    const wait_api_t *api;
    rkey_t *key;
    thread_id_t tid;
    ptr_uint_t object, aux;
    int caller_mod;
    ptr_uint_t caller_offset;
    int64 start, end, region_seq;
    uint64 begin_us, elapsed_us, sys_begin, potential_syscalls;
    ptr_int_t result;
    bool returned, excluded, exclude_scope, process_shared;
    int delay_ms;
    uint64 async_scope;
    int runtime_status;
    char *indicator, *producer, *reason;
    bool declaration_error;
} wait_record_t;
static wait_record_t *wait_records;
static void json_str(file_t f, const char *s);
static int module_index_for_pc(app_pc pc);

/* Annotation strings are copied at the checkpoint: Python argument storage may
 * disappear on return. Reject unreadable/overlong strings rather than truncate. */
static char *
wait_annotation_string(const char *source, bool *error)
{
    char buffer[2049];
    size_t n;
    if (!source) return NULL;
    for (n = 0; n < sizeof(buffer); n++) {
        if (!dr_safe_read(source + n, 1, &buffer[n], NULL)) break;
        if (!buffer[n]) {
            char *copy = dr_global_alloc(n + 1);
            memcpy(copy, buffer, n + 1);
            return copy;
        }
    }
    *error = true;
    return NULL;
}

static void
wait_pre(void *wrapcxt, void **user_data)
{
    const wait_hook_t *hook = (const wait_hook_t *)*user_data;
    const wait_api_t *api = hook->api;
    thread_t *t = cur_thread(drwrap_get_drcontext(wrapcxt));
    wait_record_t *r;
    /* GIL handoff and validation inside annotations are not application waits.
     * Retain the explicit checkpoint itself even inside this exclusion scope. */
    bool runtime = strncmp(api->kind, "runtime_", 8) == 0;
    if (t->marker_api_depth && strncmp(api->kind, "declared_", 9) != 0 && !runtime) {
        *user_data = NULL;
        return;
    }
    if (hook->excluded) {
        byte *base = dr_get_dr_segment_base(tls_seg);
        (*(ptr_uint_t *)(base + TLS_EXCLUDE_DEPTH))++;
        if (strncmp(api->kind, "declared_", 9) != 0)
            dr_atomic_add64_return_sum(&excluded_calls, 1);
    }
    *user_data = hook->excluded ? (void *)1 : NULL;
    /* Retain outer API semantics; libc and CUDA adapters often call each other. */
    if (t->wait_api_depth)
        return;
    /* Runtime housekeeping outside application regions can exceed millions of
     * mutex calls. Aggregate it, but retain semaphore/CUDA object history needed
     * to connect later in-region waits to earlier publications/initialization. */
    {
        const char *region = t->depth ? t->stack[t->depth - 1].key->region : "";
        bool application = region[0] && strncmp(region, "_perfmark", 9) != 0 &&
                           strncmp(region, "perf.", 5) != 0 && !strchr(region, ':');
        if (!application && strncmp(api->name, "sem_", 4) != 0 &&
            strncmp(api->name, "cuda", 4) != 0 &&
            strncmp(api->kind, "declared_", 9) != 0 && !runtime) {
            dr_atomic_add64_return_sum(&wait_unattributed[api - wait_apis], 1);
            return;
        }
    }
    dr_mutex_lock(wait_lock);
    if (wait_count == opt_max_wait_records) {
        wait_dropped++;
        dr_mutex_unlock(wait_lock);
        return;
    }
    r = &wait_records[wait_count++];
    r->start = ++wait_order;
    dr_mutex_unlock(wait_lock);
    r->api = api;
    if (strcmp(api->kind, "runtime_scope") == 0)
        t->async_scope = (uint64)(ptr_uint_t)drwrap_get_arg(wrapcxt, 0);
    r->async_scope = t->async_scope;
    if (strcmp(api->kind, "runtime_wait_end") == 0)
        r->runtime_status = (int)(ptr_int_t)drwrap_get_arg(wrapcxt, 2);
    /* Attribute the actual native caller. A mutex seen while a Python region
     * runs may belong to the interpreter, allocator, or a library. Preserve
     * that evidence instead of guessing ownership from the active region.
     * Resolve symbols offline; do not put symbol lookup on this hot path. */
    {
        app_pc caller = drwrap_get_retaddr(wrapcxt);
        r->caller_mod = caller ? module_index_for_pc(caller - 1) : -1;
        r->caller_offset = r->caller_mod >= 0
            ? (ptr_uint_t)(caller - 1 - modules[r->caller_mod].start) : 0;
    }
    r->exclude_scope = hook->excluded;
    r->tid = t->tid;
    if (t->depth) {
        r->key = t->stack[t->depth - 1].key;
        r->region_seq = t->stack[t->depth - 1].seq_begin;
    }
    r->object = api->object_arg < 0 ? 0 : (ptr_uint_t)drwrap_get_arg(wrapcxt, api->object_arg);
    r->aux = api->aux_arg < 0 ? 0 : (ptr_uint_t)drwrap_get_arg(wrapcxt, api->aux_arg);
    r->process_shared = strcmp(api->kind, "sem_init") == 0 &&
                        (ptr_uint_t)drwrap_get_arg(wrapcxt, 1) != 0;
    r->excluded = *(ptr_uint_t *)(dr_get_dr_segment_base(tls_seg) + TLS_EXCLUDE_DEPTH) != 0;
    if (strcmp(api->name, "perfmark_wait") == 0) {
        r->indicator = wait_annotation_string(drwrap_get_arg(wrapcxt, 2), &r->declaration_error);
        r->producer = wait_annotation_string(drwrap_get_arg(wrapcxt, 3), &r->declaration_error);
        if (!r->indicator || !r->indicator[0]) r->declaration_error = true;
    } else if (strcmp(api->name, "perfmark_wait_null") == 0) {
        r->indicator = wait_annotation_string(drwrap_get_arg(wrapcxt, 0), &r->declaration_error);
        r->reason = wait_annotation_string(drwrap_get_arg(wrapcxt, 1), &r->declaration_error);
        if (!r->indicator || !r->indicator[0] || !r->reason || !r->reason[0]) r->declaration_error = true;
    }
    r->sys_begin = t->wait_syscalls;
    t->wait_api_depth++;
    *user_data = r;
    if (opt_wait_delay_ms && r->key &&
        strcmp(api->kind, opt_wait_delay_event ? "declared_publish" : "sem_publish") == 0 &&
        strcmp(r->key->region, opt_wait_delay_region) == 0) {
        r->delay_ms = opt_wait_delay_ms;
        dr_atomic_add64_return_sum(&wait_delay_injections, 1);
        dr_sleep(opt_wait_delay_ms);
    }
    r->begin_us = dr_get_microseconds(); /* injection itself excluded from API elapsed time */
}

static void
wait_post(void *wrapcxt, void *user_data)
{
    wait_record_t *r = (wait_record_t *)user_data;
    thread_t *t;
    if (!r)
        return;
    if (user_data == (void *)1 || r->exclude_scope) {
        byte *base = dr_get_dr_segment_base(tls_seg);
        (*(ptr_uint_t *)(base + TLS_EXCLUDE_DEPTH))--;
        if (user_data == (void *)1)
            return;
    }
    t = cur_thread(dr_get_current_drcontext());
    t->wait_api_depth--;
    r->elapsed_us = dr_get_microseconds() - r->begin_us;
    r->potential_syscalls = t->wait_syscalls - r->sys_begin;
    r->returned = wrapcxt != NULL;
    if (wrapcxt) {
        r->result = (ptr_int_t)drwrap_get_retval(wrapcxt);
        if (r->result == 0 && (strcmp(r->api->kind, "event_create") == 0 ||
                               strcmp(r->api->kind, "stream_create") == 0)) {
            ptr_uint_t event;
            if (dr_safe_read((void *)r->object, sizeof(event), &event, NULL))
                r->object = event;
            else
                r->returned = false;
        }
    }
    dr_mutex_lock(wait_lock);
    r->end = ++wait_order;
    dr_mutex_unlock(wait_lock);
}

static void
wait_wrap(app_pc p, const wait_api_t *api)
{
    if (p && !drwrap_is_wrapped(p, wait_pre, wait_post)) {
        wait_hook_t *hook;
        if (wait_hooks == sizeof(wait_hook_info) / sizeof(wait_hook_info[0])) {
            dr_fprintf(STDERR, "drperf: too many synchronization hooks\n");
            dr_abort();
        }
        hook = &wait_hook_info[wait_hooks];
        hook->api = api;
        hook->address = p;
        hook->excluded = strncmp(api->kind, "declared_", 9) == 0 ||
                         strncmp(api->kind, "runtime_", 8) == 0;
        if (drwrap_wrap_ex(p, wait_pre, wait_post, hook, DRWRAP_UNWIND_ON_EXCEPTION))
            wait_hooks++;
    }
}

static void
wait_module(const module_data_t *mod, const char *name)
{
    uint i;
    if (!opt_waits || !name ||
        !(mod->start == main_module_start || strstr(name, "libc.") || strstr(name, "libpthread") ||
          strstr(name, "libcuda") || strstr(name, "gx_cuda") || strstr(name, "perfmark") ||
          (opt_exclude_cuda_module[0] && strcmp(name, opt_exclude_cuda_module) == 0)))
        return;
    for (i = 0; i < sizeof(wait_apis) / sizeof(wait_apis[0]); i++)
        wait_wrap((app_pc)dr_get_proc_address(mod->handle, wait_apis[i].name), &wait_apis[i]);
    /* glibc exports old AND current condition-wait ABIs under the same name.
     * Single-name lookup can return only the compatibility implementation.
     * Enumerate only libc/libpthread, never torch/CUDA/GX module tables. */
    if (strstr(name, "libc.") || strstr(name, "libpthread")) {
        dr_symbol_export_iterator_t *iter = dr_symbol_export_iterator_start(mod->handle);
        if (iter) {
            while (dr_symbol_export_iterator_hasnext(iter)) {
                dr_symbol_export_t *symbol = dr_symbol_export_iterator_next(iter);
                if (!symbol->name || !symbol->is_code || symbol->is_indirect_code)
                    continue;
                for (i = 0; i < sizeof(wait_apis) / sizeof(wait_apis[0]); i++) {
                    if (strcmp(symbol->name, wait_apis[i].name) == 0) {
                        wait_wrap(symbol->addr, &wait_apis[i]);
                        break;
                    }
                }
            }
            dr_symbol_export_iterator_stop(iter);
        }
    }
}

/* One callback per address keeps DRWRAP_NO_FRILLS: general wrapping holds a
 * shared drwrap lock while calling clients and would serialize delay probes.
 * Multiplex instruction exclusion inside the wait wrapper instead. */
static bool
wait_exclude(app_pc address)
{
    int i;
    for (i = 0; i < wait_hooks; i++) {
        if (wait_hook_info[i].address == address) {
            wait_hook_info[i].excluded = true;
            return true;
        }
    }
    return false;
}

static void
wait_syscall(void *drcontext, thread_t *t, int sysnum)
{
    if (!opt_waits)
        return;
    /* These are attempts, not a sleeping detector. Include excluded work. */
    if (sysnum == SYS_futex) {
        int op = (int)dr_syscall_get_param(drcontext, 1) & 127;
        if (op == 0 || op == 9) /* WAIT / WAIT_BITSET */
            t->wait_syscalls++;
    } else if (sysnum == SYS_poll || sysnum == SYS_ppoll ||
               sysnum == SYS_select || sysnum == SYS_pselect6 ||
               sysnum == SYS_epoll_wait || sysnum == SYS_epoll_pwait ||
               sysnum == SYS_recvfrom || sysnum == SYS_recvmsg)
        t->wait_syscalls++;
}

static void
write_wait_summary(file_t f)
{
    uint i;
    bool first = true;
    dr_fprintf(f, "    \"wait_unattributed\": {");
    for (i = 0; i < sizeof(wait_apis) / sizeof(wait_apis[0]); i++) {
        if (!wait_unattributed[i])
            continue;
        dr_fprintf(f, "%s", first ? "" : ",");
        first = false;
        json_str(f, wait_apis[i].name);
        dr_fprintf(f, ":{\"kind\":"); json_str(f, wait_apis[i].kind);
        dr_fprintf(f, ",\"calls\":%lld}", (long long)wait_unattributed[i]);
    }
    dr_fprintf(f, "},\n");
}

static void
write_waits(void)
{
    char path[560];
    int i;
    file_t f;
    if (!opt_waits)
        return;
    dr_snprintf(path, sizeof(path), "%s.waits", opt_out);
    f = dr_open_file(path, DR_FILE_WRITE_OVERWRITE);
    if (f == INVALID_FILE) {
        dr_fprintf(STDERR, "drperf: cannot write synchronization observations\n");
        return;
    }
    for (i = 0; i < wait_count; i++) {
        wait_record_t *r = &wait_records[i];
        dr_fprintf(f, "{\"api\":"); json_str(f, r->api->name);
        dr_fprintf(f, ",\"kind\":"); json_str(f, r->api->kind);
        dr_fprintf(f, ",\"region\":"); json_str(f, r->key ? r->key->region : "");
        dr_fprintf(f, ",\"callerModule\":");
        json_str(f, r->caller_mod >= 0 ? modules[r->caller_mod].name : "<unknown>");
        dr_fprintf(f, ",\"callerOffset\":\"%llu\"",
                   (unsigned long long)r->caller_offset);
        dr_fprintf(f, ",\"tid\":%d,\"regionSeq\":%lld,\"start\":%lld,\"end\":%lld,"
                   "\"object\":\"%llu\",\"aux\":\"%llu\",\"result\":%lld,"
                   "\"returned\":%s,\"elapsedUs\":%llu,\"potentialSyscalls\":%llu,"
                   "\"instructionExcluded\":%s,\"injectedDelayMs\":%d,\"processShared\":%s",
                   (int)r->tid, (long long)r->region_seq, (long long)r->start, (long long)r->end,
                   (unsigned long long)r->object, (unsigned long long)r->aux,
                   (long long)r->result, r->returned ? "true" : "false",
                   (unsigned long long)r->elapsed_us, (unsigned long long)r->potential_syscalls,
                   r->excluded ? "true" : "false", r->delay_ms,
                   r->process_shared ? "true" : "false");
        /* Probe planning needs timestamps only for the explicit checkpoints,
         * not every libc operation in a large Python capture. */
        if (strncmp(r->api->kind, "declared_", 9) == 0)
            dr_fprintf(f, ",\"beginUs\":%llu,\"endUs\":%llu",
                       (unsigned long long)r->begin_us,
                       (unsigned long long)(r->begin_us + r->elapsed_us));
        if (r->indicator) { dr_fprintf(f, ",\"indicator\":"); json_str(f, r->indicator); }
        if (r->producer) { dr_fprintf(f, ",\"producer\":"); json_str(f, r->producer); }
        if (r->reason) { dr_fprintf(f, ",\"reason\":"); json_str(f, r->reason); }
        if (r->declaration_error) dr_fprintf(f, ",\"declarationError\":true");
        if (r->async_scope)
            dr_fprintf(f, ",\"asyncScope\":\"%llu\"", (unsigned long long)r->async_scope);
        if (strcmp(r->api->kind, "runtime_wait_end") == 0)
            dr_fprintf(f, ",\"runtimeStatus\":%d", r->runtime_status);
        dr_fprintf(f, "}\n");
    }
    dr_close_file(f);
}
