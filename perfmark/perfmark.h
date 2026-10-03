/* perfmark: mark regions of an application for drperf.
 *
 * Outside DynamoRIO these are empty functions (cost: one PLT call).
 * Under drperf they are wrapped with drwrap and delimit counted regions.
 * Regions nest per thread. state_name/state_value describe the application
 * state the region ran under (e.g. "n_entries", 4096); pass "" and 0 if none.
 */
#ifndef PERFMARK_H
#define PERFMARK_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void perfmark_begin(const char *region, const char *state_name, int64_t state_value);
/* Several declared states: all n of them form the aggregation key,
 * so the cost formula can be derived in all of them (`drperf derive`). */
void perfmark_begin_v(const char *region, int n, const char *const *names, const int64_t *values);
void perfmark_end(const char *region);
/* Attach an extra named value to the innermost open region (recorded per
 * trigger in the trace; not part of the aggregation key). */
void perfmark_state(const char *name, const char *value);
/* Observational event checkpoints: no synchronization outside an explicit
 * drperf delay probe. Use a stable application ID plus a unique generation.
 * Publish immediately BEFORE the real release; waited only after readiness. */
/* Public passive checkpoints. Indicator is an expression over entry PCVs,
 * not a switch controlling whether the wait checkpoint is emitted.
 * producer may be NULL to use the release region recorded for this channel. */
int perfmark_release(uint64_t event, uint64_t generation);
int perfmark_wait(uint64_t event, uint64_t generation,
                  const char *indicator, const char *producer);
int perfmark_wait_null(const char *indicator, const char *reason);
/* Compatibility ABI for existing annotations/captures. */
int perfmark_event_publish(uint64_t event, uint64_t generation);
int perfmark_event_waited(uint64_t event, uint64_t generation);
/* Event-free refinement. Its indicator and human-reviewable reason live in
 * the wait-interface declaration; this marker never performs synchronization. */
int perfmark_waited_null(void);
/* Runtime observations, not dependency declarations. IDs identify logical
 * await invocations; adapters never supply a publisher. */
int perfmark_runtime_wait_begin(uint64_t id, uint64_t api);
int perfmark_runtime_wait_end(uint64_t id, uint64_t api, int status);
int perfmark_async_scope(uint64_t scope, uint64_t parent);
#ifdef __cplusplus
}
#endif
#endif
