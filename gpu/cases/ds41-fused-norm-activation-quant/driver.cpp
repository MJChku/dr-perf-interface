#include "ds41/gx_fixture.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
using namespace ds41::gx_fixture;
static void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
static float bf(std::uint16_t v){return std::bit_cast<float>(std::uint32_t(v)<<16);}
static std::uint16_t bf(float v){auto b=std::bit_cast<std::uint32_t>(v);return (b+32767+((b>>16)&1))>>16;}
static void close(float a,float b){require(std::isfinite(a)&&std::abs(a-b)<=1e-5+.008*std::abs(b),"fused BF16 independent oracle mismatch");}
static unsigned fp8(float v){const auto a=std::abs(v);double best=INFINITY;unsigned code=0;for(unsigned i=0;i<127;++i){const unsigned e=i/8,m=i%8;double x=e?std::ldexp(1+m/8.,int(e)-7):m/512.;const double distance=std::abs(a-x);if(distance<best||(distance==best&&!(i&1))){best=distance;code=i;}}return code|(std::signbit(v)?128:0);}
static Function kernel(Driver& api,Module m,const char* name){Function f;check(api.function(&f,m,name));return f;}
static void quant(const std::vector<std::uint16_t>& x,const std::vector<std::uint8_t>& payload,const std::vector<std::uint8_t>& scale,unsigned rows,unsigned dim){
    for(unsigned r=0;r<rows;++r)for(unsigned g=0;g<dim/32;++g){float maximum=0;for(unsigned d=0;d<32;++d)maximum=std::max(maximum,std::abs(bf(x[r*dim+g*32+d])));int exponent;const auto fraction=std::frexp(std::max(maximum,1e-4f)*(1.f/448),&exponent);if(fraction==.5f)--exponent;require(scale[r*(dim/32)+g]==exponent+127,"fused FP8 scale oracle mismatch");for(unsigned d=0;d<32;++d)require(payload[r*dim+g*32+d]==fp8(bf(x[r*dim+g*32+d])/std::ldexp(1.f,exponent)),"fused FP8 payload oracle mismatch");}
}
static void fixture(Driver& api,Module module,unsigned rows,unsigned dim,unsigned variant){
    constexpr unsigned guard=32;const std::uint16_t poison=0x4abc;unsigned ps=4;
    std::vector<std::uint16_t> h(rows*4*dim),weight(dim),y(rows*dim),g(rows*dim),up(rows*dim);
    std::vector<float> pre(rows*4),projection(rows*24),scale={.3f,.2f,.4f},base(24),route(rows);
    for(unsigned i=0;i<h.size();++i)h[i]=bf(variant==1?0.f:(int(i*7%101)-50)/127.f);
    for(unsigned d=0;d<dim;++d)weight[d]=bf(1+(int(d%7)-3)/128.f);
    for(unsigned i=0;i<y.size();++i){y[i]=bf((int(i*11%79)-39)/101.f);g[i]=bf((int(i%31)-15)*3.f);up[i]=bf((int(i%17)-8)*4.f);}
    for(unsigned r=0;r<rows;++r){route[r]=r==0?.25f:r==1?0.f:1.5f;for(unsigned j=0;j<4;++j)pre[r*4+j]=(j+1)/10.f;for(unsigned j=0;j<24;++j)projection[r*24+j]=(int((j*31+r*17)%137)-68)/23.f;}
    for(unsigned j=0;j<24;++j)base[j]=variant==2?(j%5==0?40.f:-40.f):(int(j%9)-4)/13.f;
    Buffer<std::uint16_t> dh(api,h),dw(api,weight),dy(api,y),dg(api,g),du(api,up),out(api,std::vector<std::uint16_t>(rows*dim+2*guard,poison)),post(api,std::vector<std::uint16_t>(rows*4*dim+2*guard,poison));
    Buffer<float> dp(api,pre),dproj(api,projection),ds(api,scale),db(api,base),dr(api,route),dc(api,std::vector<float>(rows*24+2*guard,-12345));
    Buffer<std::uint8_t> packed(api,std::vector<std::uint8_t>(rows*dim+2*guard,0xa7)),scales(api,std::vector<std::uint8_t>(rows*dim/32+2*guard,0xa7));
    auto hp=dh.ptr,wp=dw.ptr,yp=dy.ptr,gp=dg.ptr,uptr=du.ptr,pp=dp.ptr,pr=dproj.ptr,sp=ds.ptr,bp=db.ptr,rp=dr.ptr,cp=dc.ptr+guard*4,op=out.ptr+guard*2,postp=post.ptr+guard*2,qp=packed.ptr+guard,qsp=scales.ptr+guard;
    void* na[]={&hp,&pp,&wp,&op,&qp,&qsp,&rows,&dim,&ps};check(api.launch(kernel(api,module,"ds41_hc_pre_quant"),rows,1,1,dim>=512?256:32,1,1,0,nullptr,na,nullptr));check(api.sync());
    auto an=out.read();std::vector<std::uint16_t> dense(an.begin()+guard,an.end()-guard);
    for(unsigned r=0;r<rows;++r){std::vector<float> collapse(dim);long double sum=0;for(unsigned d=0;d<dim;++d){double v=0;for(unsigned j=0;j<4;++j)v+=pre[r*4+j]*double(bf(h[(r*4+j)*dim+d]));collapse[d]=bf(bf(float(v)));sum+=collapse[d]*collapse[d];}const double inverse=1/std::sqrt(double(sum/dim)+1e-20);for(unsigned d=0;d<dim;++d)close(bf(dense[r*dim+d]),bf(bf(float(collapse[d]*inverse*bf(weight[d])))));}
    auto p=packed.read(),s=scales.read();quant(dense,{p.begin()+guard,p.end()-guard},{s.begin()+guard,s.end()-guard},rows,dim);
    require(std::all_of(an.begin(),an.begin()+guard,[&](auto v){return v==poison;})&&std::all_of(an.end()-guard,an.end(),[&](auto v){return v==poison;}),"fused pre output guards");
    for(unsigned inplace:{0u,1u}){auto source=inplace?op:yp;unsigned stride=dim;auto before=inplace?dense:y;void* ra[]={&source,&wp,&op,&qp,&qsp,&rows,&dim,&stride,&stride};check(api.launch(kernel(api,module,"ds41_rms_quant"),rows,1,1,dim>=512?256:32,1,1,0,nullptr,ra,nullptr));check(api.sync());auto values=out.read();dense.assign(values.begin()+guard,values.end()-guard);for(unsigned r=0;r<rows;++r){double sum=0;for(unsigned d=0;d<dim;++d){double v=bf(before[r*dim+d]);sum+=v*v;}const double inverse=1/std::sqrt(sum/dim+1e-20);for(unsigned d=0;d<dim;++d)close(bf(dense[r*dim+d]),bf(bf(float(bf(before[r*dim+d])*bf(weight[d])*inverse))));}p=packed.read();s=scales.read();quant(dense,{p.begin()+guard,p.end()-guard},{s.begin()+guard,s.end()-guard},rows,dim);}
    void* ca[]={&hp,&pr,&sp,&bp,&yp,&cp,&postp,&rows,&dim};check(api.launch(kernel(api,module,"ds41_hc_coeff_post"),rows,1,1,dim>=512?256:32,1,1,0,nullptr,ca,nullptr));check(api.sync());auto actual=dc.read();auto mixed=post.read();
    for(unsigned r=0;r<rows;++r){long double sq=0;for(unsigned d=0;d<4*dim;++d){double v=bf(h[r*4*dim+d]);sq+=v*v;}double inv=1/std::sqrt(double(sq/(4*dim))+1e-20),c[24],matrix[4][4];
        for(unsigned j=0;j<4;++j){c[j]=1/(1+std::exp(-(projection[r*24+j]*inv*scale[0]+base[j])))+1e-6;c[4+j]=2/(1+std::exp(-(projection[r*24+4+j]*inv*scale[1]+base[4+j])));double maximum=-INFINITY;for(unsigned k=0;k<4;++k){matrix[j][k]=projection[r*24+8+j*4+k]*inv*scale[2]+base[8+j*4+k];maximum=std::max(maximum,matrix[j][k]);}double total=0;for(auto& v:matrix[j]){v=std::exp(v-maximum);total+=v;}for(auto& v:matrix[j])v=v/total+1e-6;}
        for(unsigned iteration=0;iteration<20;++iteration){if(iteration)for(auto& row:matrix){double total=1e-6;for(auto v:row)total+=v;for(auto& v:row)v/=total;}for(unsigned k=0;k<4;++k){double total=1e-6;for(unsigned j=0;j<4;++j)total+=matrix[j][k];for(unsigned j=0;j<4;++j)matrix[j][k]/=total;}}
        for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k)c[8+j*4+k]=matrix[j][k];for(unsigned j=0;j<24;++j)require(std::isfinite(actual[guard+r*24+j])&&std::abs(actual[guard+r*24+j]-c[j])<=2e-6+2e-5*std::abs(c[j]),"fused coefficients independent oracle");
        for(unsigned j=0;j<4;++j)for(unsigned d=0;d<dim;++d){double v=c[4+j]*bf(y[r*dim+d]);for(unsigned k=0;k<4;++k)v+=c[8+k*4+j]*bf(h[(r*4+k)*dim+d]);close(bf(mixed[guard+(r*4+j)*dim+d]),bf(bf(float(v))));}}
    for(unsigned weighted:{0u,1u}){void* sa[]={&gp,&uptr,&rp,&op,&qp,&qsp,&rows,&dim,&weighted};check(api.launch(kernel(api,module,"ds41_swiglu_quant"),(rows*(dim/32)+3)/4,1,1,128,1,1,0,nullptr,sa,nullptr));check(api.sync());auto values=out.read();dense.assign(values.begin()+guard,values.end()-guard);for(unsigned i=0;i<rows*dim;++i){double a=std::min(double(bf(g[i])),10.),b=std::clamp(double(bf(up[i])),-10.,10.);close(bf(dense[i]),bf(bf(float(a/(1+std::exp(-a))*b*(weighted?route[i/dim]:1)))));}p=packed.read();s=scales.read();quant(dense,{p.begin()+guard,p.end()-guard},{s.begin()+guard,s.end()-guard},rows,dim);}
    require(dh.read()==h&&dw.read()==weight&&dg.read()==g&&du.read()==up,"fusion input mutated");
}
int main(int argc,char** argv){try{const bool native=prepare_execution(argc,argv);std::ifstream f(argv[1]);std::string ptx{std::istreambuf_iterator<char>(f),{}};Driver api;check(api.init(0));Context ctx;check(api.create(&ctx,0,0));Module mod;check(api.module(&mod,ptx.c_str()));unsigned cases=0;for(unsigned dim:{32u,288u,512u,5120u})for(unsigned variant:{0u,1u,2u}){fixture(api,mod,3,dim,variant);++cases;}check(api.unload(mod));check(api.destroy(ctx));std::cout<<"{\"numerical_status\":\"pass\",\"kernels\":4,\"cases\":"<<cases<<",\"oracle\":\"independent double HC/Sinkhorn/SwiGLU and exhaustive FP8 codes\",\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
