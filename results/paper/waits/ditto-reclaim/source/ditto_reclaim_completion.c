/* Controlled completion fence for the Ditto control-path regression.
 * No GPU computation or compression is performed by this fixture. */
#include <assert.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>
#include "perfmark.h"
static sem_t fence;
static pthread_t worker;
static atomic_int done;
static uint64_t generation;
static void *complete(void *unused) {
    (void)unused;
    usleep(100000);
    perfmark_begin("ditto.compression.complete", "generation", generation);
    perfmark_event_publish(7001,generation);
    atomic_store(&done,1);
    assert(sem_post(&fence)==0);
    perfmark_end("ditto.compression.complete");
    return NULL;
}
void completion_start(uint64_t g) {
    generation=g;atomic_store(&done,0);
    assert(sem_init(&fence,0,0)==0);
    assert(pthread_create(&worker,NULL,complete,NULL)==0);
}
void completion_wait(void) {assert(sem_wait(&fence)==0);}
int completion_ready(void) {return atomic_load(&done);}
void completion_finish(void) {
    assert(pthread_join(worker,NULL)==0);
    assert(sem_destroy(&fence)==0);
}
