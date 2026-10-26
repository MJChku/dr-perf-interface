#include <dlfcn.h>
#include <cstdio>
#include <typeinfo>
#include <stdexcept>
#include <cstdlib>
extern "C" void __cxa_throw(void* ex, std::type_info* t, void(*dest)(void*)) {
 bool show=*t==typeid(std::runtime_error)||*t==typeid(std::invalid_argument)||*t==typeid(std::out_of_range);if(show)std::fprintf(stderr,"CPU_THROW type=%s",t->name());
 if(*t==typeid(std::runtime_error)||*t==typeid(std::invalid_argument)||*t==typeid(std::out_of_range))std::fprintf(stderr," what=%s",static_cast<std::exception*>(ex)->what());
 if(show)std::fprintf(stderr,"\n");using Fn=void(*)(void*,std::type_info*,void(*)(void*));static auto f=(Fn)dlsym(RTLD_NEXT,"__cxa_throw");f(ex,t,dest);std::abort();
}
