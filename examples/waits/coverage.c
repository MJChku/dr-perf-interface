/* Shared B has no event parameter. Caller checkpoints cover its native work. */
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "perfmark.h"
static sem_t tokens;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
static void B(int retries) {
    perfmark_begin("B", "retries", retries);
    if (retries) {
        assert(pthread_mutex_lock(&lock)==0);
        for (int i=0;i<retries;++i) {
            struct timespec past={0,0};
            assert(pthread_cond_timedwait(&changed,&lock,&past)==ETIMEDOUT);
        }
        assert(pthread_mutex_unlock(&lock)==0);
    }
    assert(sem_wait(&tokens)==0);
    perfmark_end("B");
}
static void nested(int depth) {
    if (!depth) { B(3); return; }
    perfmark_begin("wrapper", "depth", depth);
    nested(depth-1);
    perfmark_end("wrapper");
}
static void publish(const char *name, uint64_t event, uint64_t generation) {
    perfmark_begin(name, "generation", generation);
    perfmark_event_publish(event,generation);
    for (int i=0;i<32;++i) assert(sem_post(&tokens)==0);
    perfmark_end(name);
}
static void *worker(void *p) {
    uint64_t id=(uintptr_t)p;
    const char *name=id==1?"A":"C";
    perfmark_begin(name,"n",1);
    B(0);
    perfmark_event_waited(id,1);
    perfmark_end(name);
    return NULL;
}
int main(int argc,char **argv) {
    const char *mode=argc>1?argv[1]:"deep";
    perfmark_begin("capture","n",1);perfmark_end("capture");
    if (!strcmp(mode,"child_publisher")) {
        perfmark_begin("A","n",1);
        perfmark_begin("B","n",1);
        perfmark_event_publish(1,1);
        assert(pthread_mutex_lock(&lock)==0);
        assert(pthread_mutex_unlock(&lock)==0);
        perfmark_end("B");
        perfmark_event_waited(1,1);
        perfmark_end("A");
        return 0;
    }
    if (!strcmp(mode,"null_shared") || !strcmp(mode,"null_parent")) {
        int value = 0;
        int parent_marker = !strcmp(mode,"null_parent");
        for (int i=0; i<2; ++i) {
            const char *parent = i ? "C" : "A";
            perfmark_begin(parent,"need",1);
            for (int need=0; need<=1; ++need) {
                perfmark_begin("B","need",need);
                if (need) {
                    assert(pthread_mutex_lock(&lock)==0);
                    ++value;
                    assert(pthread_mutex_unlock(&lock)==0);
                    if (!parent_marker) perfmark_waited_null();
                }
                perfmark_end("B");
            }
            if (parent_marker) perfmark_waited_null();
            perfmark_end(parent);
        }
        assert(value==2);
        return 0;
    }
    if (!strcmp(mode,"uncontended")) {
        /* There is no publishing peer: the program starts no other thread,
         * and this private mutex starts unlocked. A conservative API budget
         * still flags the acquisition. Do not fabricate a waited declaration
         * just to clear it: this documents the detector's scope boundary. */
        pthread_mutex_t private_lock;
        assert(pthread_mutex_init(&private_lock,NULL)==0);
        perfmark_begin("A","n",1);
        assert(pthread_mutex_lock(&private_lock)==0);
        assert(pthread_mutex_unlock(&private_lock)==0);
        perfmark_end("A");
        assert(pthread_mutex_destroy(&private_lock)==0);
        return 0;
    }
    assert(sem_init(&tokens,0,0)==0);
    publish("P",1,1);
    if (!strcmp(mode,"parallel")) {
        publish("Q",2,1);
        pthread_t a,c;
        assert(pthread_create(&a,NULL,worker,(void *)(uintptr_t)1)==0);
        assert(pthread_create(&c,NULL,worker,(void *)(uintptr_t)2)==0);
        assert(pthread_join(a,NULL)==0);assert(pthread_join(c,NULL)==0);
    } else if (!strcmp(mode,"shared") || !strcmp(mode,"relocated")) {
        int wrong=!strcmp(mode,"shared");
        for (int i=0;i<2;++i) {
            if (i) publish("Q",wrong?1:2,wrong?2:1);
            const char *name=i?"C":"A";
            perfmark_begin(name,"n",1);
            if (wrong) {
                perfmark_begin("B","n",1);
                assert(sem_wait(&tokens)==0);
                perfmark_event_waited(1,i+1);
                perfmark_end("B");
            } else {
                B(0);perfmark_event_waited(i+1,1);
            }
            perfmark_end(name);
        }
    } else if (!strcmp(mode,"branch")) {
        for(int i=0;i<6;++i) {
            perfmark_begin("A","need",i%2);
            if(i%2) {B(2);perfmark_event_waited(1,1);}
            perfmark_end("A");
        }
    } else {
        perfmark_begin("A","n",1);
        if (!strcmp(mode,"deep")) {nested(8);perfmark_event_waited(1,1);}
        else if (!strcmp(mode,"retries")) {B(8);perfmark_event_waited(1,1);}
        else if (!strcmp(mode,"direct_missing") || !strcmp(mode,"direct_two")) {
            /* Two successes at the same object/site are two obligations. */
            for (int i=0;i<2;++i) assert(sem_wait(&tokens)==0);
            perfmark_event_waited(1,1);
            if (!strcmp(mode,"direct_two")) perfmark_event_waited(1,1);
        }
        else if (!strcmp(mode,"missing") || !strcmp(mode,"two")) {
            B(0);B(0);perfmark_event_waited(1,1);
            if (!strcmp(mode,"two")) perfmark_event_waited(1,1);
        } else if (!strcmp(mode,"early")) {perfmark_event_waited(1,1);B(0);}
        else if (!strcmp(mode,"own")) {
            assert(sem_wait(&tokens)==0);B(0);
            perfmark_event_waited(1,1);perfmark_event_waited(1,1);
        } else if (!strcmp(mode,"sibling")) {
            B(0);perfmark_begin("C","n",1);
            perfmark_event_waited(1,1);perfmark_end("C");
        } else if (!strcmp(mode,"none")) {B(0);B(0);}
        else return 2;
        perfmark_end("A");
    }
    assert(sem_destroy(&tokens)==0);
    return 0;
}
