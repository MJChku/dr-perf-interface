/* drperf_attach: start DynamoRIO partway through a process's life.
 *
 * Preloaded (LD_PRELOAD) so libdynamorio is mapped and its address space
 * reserved before the application runs, but nothing is translated until
 * drperf_attach_now() is called.  The marker library calls it at the first
 * region, so everything before that (imports, model loading, warm-up) runs
 * natively.  DynamoRIO takes over the threads that already exist.
 */
extern int dr_app_setup_and_start(void);

__attribute__((visibility("default"))) int
drperf_attach_now(void)
{
    return dr_app_setup_and_start();
}
