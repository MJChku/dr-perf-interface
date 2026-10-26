#include "ds41/gx_fixture.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
using namespace ds41::gx_fixture;
static void require(bool v,const char* s) { if(!v)throw std::runtime_error(s); }
static std::uint16_t bf(float v) {auto b=std::bit_cast<std::uint32_t>(v);return (b+32767+((b>>16)&1))>>16;}
static float bf(std::uint16_t v) {return std::bit_cast<float>(std::uint32_t(v)<<16);}
static void close(float a,float b,const char* s) {require(std::isfinite(a)&&std::abs(a-b)<=1e-5+0.008*std::abs(b),s);}
static unsigned normalization_threads(unsigned dim){
#ifdef DS41_WIDE_NORMALIZATION
    return dim>=512?256:32;
#else
    return 32;
#endif
}
static Function function(Driver& api,Module mod,const char* name) {Function f;check(api.function(&f,mod,name));return f;}

static Module module_for_splitk;
static void dense(Driver& api,Function fn,unsigned flags,unsigned variant=0,bool long_k=false,bool production=false,unsigned requested_splits=8) {
    unsigned m=production?3:long_k?1:variant>=3?67:3,n=production?24:long_k?5:variant>=3?24:17,k=production?20480:long_k?4099:35,as=k+3,bs=k+5,os=n+7;
    constexpr unsigned guard=64;
    const unsigned ab=flags&1 ? 4:2,bb=flags&2 ? 4:2,ob=flags&4 ? 4:2;
    std::vector<std::uint8_t> a(guard*2+m*as*ab,0xA7),b(guard*2+n*bs*bb,0xA7),expected(guard*2+m*os*ob,0xA7);
    const auto initial=expected;
    auto av=[&](unsigned r,unsigned j){float x=long_k?std::sin(float(r*7+j*3)*.17f)*.123f:(int((r*7+j*3)%19)-9)/32.0f;return flags&1?x:bf(bf(x));};
    auto bv=[&](unsigned c,unsigned j){float x=long_k?std::cos(float(c*11+j*5)*.11f)*.067f:(int((c*11+j*5)%23)-11)/64.0f;return flags&2?x:bf(bf(x));};
    for(unsigned r=0;r<m;++r)for(unsigned j=0;j<k;++j){float x=av(r,j);auto v=bf(x);
        if(flags&1)std::memcpy(a.data()+guard+(r*as+j)*ab,&x,4);else std::memcpy(a.data()+guard+(r*as+j)*ab,&v,2);}
    for(unsigned c=0;c<n;++c)for(unsigned j=0;j<k;++j){float x=bv(c,j);auto v=bf(x);
        if(flags&2)std::memcpy(b.data()+guard+(c*bs+j)*bb,&x,4);else std::memcpy(b.data()+guard+(c*bs+j)*bb,&v,2);}
    for(unsigned r=0;r<m;++r)for(unsigned c=0;c<n;++c){double sum=0;
        for(unsigned j=0;j<k;++j)sum+=double(av(r,j))*bv(c,j);
        const float result=sum;const auto code=bf(result);
        if(flags&4)std::memcpy(expected.data()+guard+(r*os+c)*ob,&result,4);else std::memcpy(expected.data()+guard+(r*os+c)*ob,&code,2);}
    Buffer<std::uint8_t> da(api,a),db(api,b),out(api,initial);
    auto ap=da.ptr+guard,bp=db.ptr+guard,op=out.ptr+guard;
    void* args[]={&ap,&bp,&op,&m,&n,&k,&flags,&as,&bs,&os};
    if(variant==5){
        unsigned splits=k%256==0?requested_splits:1,ps=os;Buffer<float> partial(api,std::vector<float>(static_cast<std::size_t>(splits)*m*ps+32,-719));auto pp=partial.ptr;
        void* ka[]={&ap,&bp,&pp,&m,&n,&k,&flags,&as,&bs,&ps,&splits};
        check(api.launch(fn,(m+63)/64,(n+31)/32,splits,128,1,1,0,nullptr,ka,nullptr));
        auto reduce=function(api,module_for_splitk,"ds41_hc_splitk_reduce");void* ra[]={&pp,&op,&m,&n,&ps,&os,&splits};
        check(api.launch(reduce,(m*n+127)/128,1,1,128,1,1,0,nullptr,ra,nullptr));
    }
    else if(variant==4){check(api.launch(fn,(m+63)/64,(n+31)/32,1,128,1,1,0,nullptr,args,nullptr));}
    else if(variant==3){check(api.launch(fn,(m+31)/32,(n+31)/32,1,128,1,1,0,nullptr,args,nullptr));}
    else check(api.launch(fn,variant==1?(n+3)/4:variant==2?(n+63)/64:(n+15)/16,variant==1?m:variant==2?(m+31)/32:(m+15)/16,1,variant==1||variant==2?128:16,variant==1||variant==2?1:16,1,0,nullptr,args,nullptr));check(api.sync());
    auto actual=out.read();
    if(long_k)for(unsigned r=0;r<m;++r)for(unsigned c=0;c<n;++c){const auto offset=guard+(r*os+c)*ob;float x,y;
        if(flags&4){std::memcpy(&x,actual.data()+offset,4);std::memcpy(&y,expected.data()+offset,4);require(std::isfinite(x)&&std::abs(x-y)<=1e-6+3e-5*std::abs(y),"dense FP32 long-K oracle mismatch");}
        else {std::uint16_t xx,yy;std::memcpy(&xx,actual.data()+offset,2);std::memcpy(&yy,expected.data()+offset,2);close(bf(xx),bf(yy),"dense BF16 long-K oracle mismatch");}
        std::memcpy(actual.data()+offset,expected.data()+offset,ob);}
    require(actual==expected,"dense dtype/stride/tail/guard mismatch");require(da.read()==a&&db.read()==b,"dense input changed");
}

static void hc(Driver& api,Module mod,unsigned dim,bool extremes) {
    unsigned rows=2,pre_stride=4;constexpr unsigned guard=32;
    std::vector<std::uint16_t> h(rows*4*dim+guard*2,0x4abc),weights(dim+guard*2,0x4abc);
    std::vector<float> projection(rows*24+guard*2,-12345),scale(3+guard*2,-12345),base(24+guard*2,-12345),previous(rows*4+guard*2,-12345);
    for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<4*dim;++j)h[guard+r*4*dim+j]=bf(r ? 0.0f:float(int((j*7)%101)-50)/127);
    for(unsigned d=0;d<dim;++d)weights[guard+d]=bf(1+float(int(d%7)-3)/128);
    for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<24;++j)projection[guard+r*24+j]=float(int((j*31+r*17)%137)-68)/23;
    scale[guard]=.3f;scale[guard+1]=.2f;scale[guard+2]=.4f;
    for(unsigned j=0;j<24;++j)base[guard+j]=extremes ? ((j%5==0) ? 40.0f:-40.0f):float(int(j%9)-4)/13;
    for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<4;++j)previous[guard+r*4+j]=float(j+1)/10;
    std::vector<float> expected(rows*24+guard*2,-12345);
    std::vector<std::uint16_t> norm(rows*dim+guard*2,0x4abc),post(rows*4*dim+guard*2,0x4abc);
    const auto norm_initial=norm,post_initial=post;const auto coeff_initial=expected;
    for(unsigned r=0;r<rows;++r){
        long double squares=0;for(unsigned j=0;j<4*dim;++j){auto v=bf(h[guard+r*4*dim+j]);squares+=v*v;}
        const double rs=1/std::sqrt(double(squares/(4*dim))+1e-20);
        double matrix[4][4];
        for(unsigned j=0;j<4;++j){
            expected[guard+r*24+j]=1/(1+std::exp(-(projection[guard+r*24+j]*rs*scale[guard]+base[guard+j])))+1e-6;
            expected[guard+r*24+4+j]=2/(1+std::exp(-(projection[guard+r*24+j+4]*rs*scale[guard+1]+base[guard+j+4])));
            double maximum=-INFINITY;for(unsigned k=0;k<4;++k){matrix[j][k]=projection[guard+r*24+8+j*4+k]*rs*scale[guard+2]+base[guard+8+j*4+k];maximum=std::max(maximum,matrix[j][k]);}
            double sum=0;for(unsigned k=0;k<4;++k){matrix[j][k]=std::exp(matrix[j][k]-maximum);sum+=matrix[j][k];}
            for(unsigned k=0;k<4;++k)matrix[j][k]=matrix[j][k]/sum+1e-6;
        }
        for(unsigned iteration=0;iteration<20;++iteration){
            if(iteration)for(auto& row:matrix){double sum=1e-6;for(auto v:row)sum+=v;for(auto& v:row)v/=sum;}
            for(unsigned k=0;k<4;++k){double sum=1e-6;for(unsigned j=0;j<4;++j)sum+=matrix[j][k];for(unsigned j=0;j<4;++j)matrix[j][k]/=sum;}
        }
        for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k)expected[guard+r*24+8+j*4+k]=matrix[j][k];
        std::vector<float> collapsed(dim);long double variance=0;
        for(unsigned d=0;d<dim;++d){double sum=0;for(unsigned j=0;j<4;++j)sum+=previous[guard+r*4+j]*double(bf(h[guard+(r*4+j)*dim+d]));collapsed[d]=bf(bf(float(sum)));variance+=collapsed[d]*collapsed[d];}
        const auto inverse=1/std::sqrt(double(variance/dim)+1e-20);
        for(unsigned d=0;d<dim;++d)norm[guard+r*dim+d]=bf(float(collapsed[d]*inverse*bf(weights[guard+d])));
        for(unsigned j=0;j<4;++j)for(unsigned d=0;d<dim;++d){
            double value=expected[guard+r*24+4+j]*double(bf(norm[guard+r*dim+d]));
            for(unsigned k=0;k<4;++k)value+=expected[guard+r*24+8+k*4+j]*double(bf(h[guard+(r*4+k)*dim+d]));
            post[guard+(r*4+j)*dim+d]=bf(float(value));
        }
    }
    Buffer<std::uint16_t> dh(api,h),dw(api,weights),dn(api,norm_initial),dp(api,post_initial);
    Buffer<float> dm(api,projection),ds(api,scale),db(api,base),dc(api,coeff_initial),dpre(api,previous);
    auto hp=dh.ptr+guard*2,wp=dw.ptr+guard*2,np=dn.ptr+guard*2,pp=dp.ptr+guard*2;
    auto mp=dm.ptr+guard*4,sp=ds.ptr+guard*4,bp=db.ptr+guard*4,cp=dc.ptr+guard*4,pre=dpre.ptr+guard*4;
    auto cf=function(api,mod,"ds41_hc_coefficients"),nf=function(api,mod,"ds41_hc_pre_norm"),pf=function(api,mod,"ds41_hc_post");
    void* ca[]={&hp,&mp,&sp,&bp,&cp,&rows,&dim};check(api.launch(cf,rows,1,1,normalization_threads(dim),1,1,0,nullptr,ca,nullptr));
    void* na[]={&hp,&pre,&wp,&np,&rows,&dim,&pre_stride};check(api.launch(nf,rows,1,1,normalization_threads(dim),1,1,0,nullptr,na,nullptr));
    void* pa[]={&np,&hp,&cp,&pp,&rows,&dim};check(api.launch(pf,(rows*4*dim+127)/128,1,1,128,1,1,0,nullptr,pa,nullptr));check(api.sync());
    auto coeff=dc.read();for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<24;++j){const auto i=guard+r*24+j;require(std::isfinite(coeff[i])&&std::abs(coeff[i]-expected[i])<=2e-6+2e-5*std::abs(expected[i]),"HC coefficient oracle mismatch");coeff[i]=expected[i];}
    require(coeff==expected,"HC coefficient guards changed");
    auto an=dn.read(),ap=dp.read();
    for(unsigned r=0;r<rows;++r){for(unsigned d=0;d<dim;++d){auto i=guard+r*dim+d;close(bf(an[i]),bf(norm[i]),"HC pre-normalization mismatch");an[i]=norm[i];}
        for(unsigned j=0;j<4*dim;++j){auto i=guard+r*4*dim+j;close(bf(ap[i]),bf(post[i]),"HC transpose-oriented residual mixing mismatch");ap[i]=post[i];}}
    require(an==norm&&ap==post,"HC output guards changed");require(dh.read()==h&&dm.read()==projection&&ds.read()==scale&&db.read()==base&&dpre.read()==previous,"HC input mutated");
}

static void rms(Driver& api,Function fn,unsigned dim) {
    unsigned rows=3,xs=dim+7,os=dim+11;constexpr unsigned guard=32;
    std::vector<std::uint16_t> x(rows*xs+guard*2,0x4abc),weight(dim+guard*2,0x4abc),expected(rows*os+guard*2,0x4abc);
    for(unsigned d=0;d<dim;++d)weight[guard+d]=bf(1+float(d%7)/128);
    for(unsigned r=0;r<rows;++r){double sum=0;for(unsigned d=0;d<dim;++d){x[guard+r*xs+d]=bf(r==0 ? 0.0f:r==1 ? 1e-20f:float(int(d%13)-6)/17);auto v=bf(x[guard+r*xs+d]);sum+=double(v)*v;}
        for(unsigned d=0;d<dim;++d)expected[guard+r*os+d]=bf(float(bf(weight[guard+d])*double(bf(x[guard+r*xs+d]))/std::sqrt(sum/dim+1e-20)));}
    auto initial=std::vector<std::uint16_t>(expected.size(),0x4abc);Buffer<std::uint16_t> dx(api,x),dw(api,weight),out(api,initial);
    auto xp=dx.ptr+guard*2,wp=dw.ptr+guard*2,op=out.ptr+guard*2;void* args[]={&xp,&wp,&op,&rows,&dim,&xs,&os};
    check(api.launch(fn,rows,1,1,normalization_threads(dim),1,1,0,nullptr,args,nullptr));check(api.sync());auto actual=out.read();
    for(unsigned r=0;r<rows;++r)for(unsigned d=0;d<dim;++d){auto i=guard+r*os+d;close(bf(actual[i]),bf(expected[i]),"RMS oracle mismatch");actual[i]=expected[i];}
    require(actual==expected,"RMS guards changed");require(dx.read()==x&&dw.read()==weight,"RMS input changed");
}

static void swiglu(Driver& api,Function fn,unsigned weighted) {
    unsigned rows=2,dim=67;constexpr unsigned guard=32;
    std::vector<std::uint16_t> g(rows*dim+guard*2,0x4abc),u=g,expected=g;std::vector<float> weight{.13f,.71f};
    for(unsigned i=0;i<rows*dim;++i){g[guard+i]=bf(float(int(i%31)-15)*3);u[guard+i]=bf(float(int(i%17)-8)*4);
        double a=std::min(double(bf(g[guard+i])),10.0),b=std::clamp(double(bf(u[guard+i])),-10.0,10.0);
        expected[guard+i]=bf(float(a/(1+std::exp(-a))*b*(weighted ? weight[i/dim]:1)));}
    Buffer<std::uint16_t> dg(api,g),du(api,u),out(api,std::vector<std::uint16_t>(g.size(),0x4abc));Buffer<float> dw(api,weight);
    auto gp=dg.ptr+guard*2,up=du.ptr+guard*2,op=out.ptr+guard*2,wp=dw.ptr;void* args[]={&gp,&up,&wp,&op,&rows,&dim,&weighted};
    check(api.launch(fn,(rows*dim+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));check(api.sync());auto actual=out.read();
    for(unsigned i=guard;i<guard+rows*dim;++i){close(bf(actual[i]),bf(expected[i]),"weighted/clamped SwiGLU mismatch");actual[i]=expected[i];}
    require(actual==expected,"SwiGLU guards changed");require(dg.read()==g&&du.read()==u&&dw.read()==weight,"SwiGLU inputs changed");
}

static void rope(Driver& api,Function fn,unsigned inverse) {
    unsigned rows=2,heads=2,dim=64,rd=32,stride=heads*dim+7;constexpr unsigned guard=32;
    std::vector<std::uint16_t> input(rows*stride+guard*2,0x4abc);
    std::vector<float> frequency(18*rd);std::vector<std::uint32_t> positions{3,17};
    for(unsigned p=0;p<18;++p)for(unsigned j=0;j<rd/2;++j){frequency[p*rd+j*2]=std::cos(.01f*(p+1)*(j+1));frequency[p*rd+j*2+1]=std::sin(.01f*(p+1)*(j+1));}
    for(unsigned r=0;r<rows;++r)for(unsigned d=0;d<heads*dim;++d)input[guard+r*stride+d]=bf(float(int((d*3+r*13)%101)-50)/43);
    auto expected=input;
    for(unsigned r=0;r<rows;++r)for(unsigned h=0;h<heads;++h)for(unsigned j=0;j<rd/2;++j){const auto i=guard+r*stride+h*dim+dim-rd+j*2;
        double a=bf(input[i]),b=bf(input[i+1]),c=frequency[positions[r]*rd+j*2],s=frequency[positions[r]*rd+j*2+1]*(inverse ? -1:1);
        expected[i]=bf(float(a*c-b*s));expected[i+1]=bf(float(a*s+b*c));}
    Buffer<std::uint16_t> dx(api,input);Buffer<float> df(api,frequency);Buffer<std::uint32_t> dp(api,positions);
    auto xp=dx.ptr+guard*2,fp=df.ptr,pp=dp.ptr;void* args[]={&xp,&fp,&pp,&rows,&heads,&dim,&rd,&stride,&inverse};
    check(api.launch(fn,(rows*heads*(rd/2)+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));check(api.sync());auto actual=dx.read();
    for(unsigned r=0;r<rows;++r)for(unsigned h=0;h<heads;++h)for(unsigned j=dim-rd;j<dim;++j){auto i=guard+r*stride+h*dim+j;close(bf(actual[i]),bf(expected[i]),"RoPE absolute position/inverse mismatch");actual[i]=expected[i];}
    require(actual==expected,"RoPE changed prefix/stride/guards");require(df.read()==frequency&&dp.read()==positions,"RoPE metadata changed");
}
int main(int argc,char** argv) {
    try {const bool native=prepare_execution(argc,argv);std::ifstream file(argv[1]);require(file.good(),"PTX missing");
        std::string ptx{std::istreambuf_iterator<char>(file),{}};Driver api;check(api.init(0));Context ctx;check(api.create(&ctx,0,0));Module mod;check(api.module(&mod,ptx.c_str()));
#ifdef DS41_HC_SPLITK_ONLY
        module_for_splitk=mod;
#ifdef DS41_DRPERF_HC_SLICE
        dense(api,function(api,mod,"ds41_hc_splitk"),7,5,true,true,128);
#else
        dense(api,function(api,mod,"ds41_hc_splitk"),7,5,false);dense(api,function(api,mod,"ds41_hc_splitk"),7,5,true);for(unsigned parts:{8u,32u,64u,128u})dense(api,function(api,mod,"ds41_hc_splitk"),7,5,true,true,parts);
        #endif
        check(api.unload(mod));check(api.destroy(ctx));std::cout<<"{\"numerical_status\":\"pass\",\"scope\":\"parallel K-partition HC adapter and reduction versus independent FP32 oracle\",\"fixtures\":"<<
#ifdef DS41_DRPERF_HC_SLICE
1
#else
6
#endif
<<",\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";return 0;
#endif
#ifdef DS41_HC_TF32X3_ONLY
        dense(api,function(api,mod,"ds41_hc_tf32x3"),7,4,false);dense(api,function(api,mod,"ds41_hc_tf32x3"),7,4,true);dense(api,function(api,mod,"ds41_hc_tf32x3"),7,4,true,true);
        check(api.unload(mod));check(api.destroy(ctx));std::cout<<"{\"numerical_status\":\"pass\",\"scope\":\"CUTLASS three-term TF32 FP32-accuracy adapter\",\"fixtures\":3,\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";return 0;
#endif
#ifdef DS41_HC_PROJECTION_ONLY
        dense(api,function(api,mod,"ds41_hc_projection"),6,3,false);dense(api,function(api,mod,"ds41_hc_projection"),6,3,true);dense(api,function(api,mod,"ds41_hc_projection"),6,3,true,true);
        check(api.unload(mod));check(api.destroy(ctx));std::cout<<"{\"numerical_status\":\"pass\",\"scope\":\"CUTLASS mixed BF16/FP32 projection adapter\",\"fixtures\":3,\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";return 0;
#endif
        for(unsigned flags=0;flags<8;++flags){
            dense(api,function(api,mod,"ds41_dense_gemm"),flags);
            dense(api,function(api,mod,"ds41_dense_tiled"),flags,0);dense(api,function(api,mod,"ds41_dense_tiled"),flags,0,true);
            dense(api,function(api,mod,"ds41_dense_matvec"),flags,1);dense(api,function(api,mod,"ds41_dense_matvec"),flags,1,true);
            if(!(flags&3)){dense(api,function(api,mod,"ds41_dense_bf16"),flags,2);dense(api,function(api,mod,"ds41_dense_bf16"),flags,2,true);}
        }
        hc(api,mod,32,false);hc(api,mod,64,true);hc(api,mod,33,false);hc(api,mod,5120,false);rms(api,function(api,mod,"ds41_rms_norm"),33);rms(api,function(api,mod,"ds41_rms_norm"),64);rms(api,function(api,mod,"ds41_rms_norm"),5120);
        swiglu(api,function(api,mod,"ds41_swiglu"),0);swiglu(api,function(api,mod,"ds41_swiglu"),1);
        rope(api,function(api,mod,"ds41_rope"),0);rope(api,function(api,mod,"ds41_rope"),1);
        check(api.unload(mod));check(api.destroy(ctx));std::cout<<"{\"numerical_status\":\"pass\",\"kernel_entries\":10,\"fixtures\":55,\"physical_gpu_used\":"<<(native ? "true":"false")<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
