#include "drperf_gpu.h"
#include "src4f5/include/latent_cache.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
using drperf::gpu::Module;using drperf::gpu::Region;using glm53::PagedFp8LayerView;
struct GatherPlan {int requests=0;int request[4]={};int offset[4]={};int count[4]={};int base_of_view[4]={};};
static void ck(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
struct Fixture {
 int T,w,P,phys,D,L,R,pt;std::vector<int>tables,map,pos,err;std::vector<uint64_t>epochs;
 std::vector<uint8_t> latent,rope,index;std::vector<PagedFp8LayerView> views;
 Fixture(int t,int rows,int dim=128,int ld=128,int rd=64,int page_tokens=256):T(t),w(rows),P((t+page_tokens)/page_tokens+1),phys(P*rows),D(dim),L(ld),R(rd),pt(page_tokens),tables(P*rows),map(rows),pos(rows,t-1),err(rows),epochs(phys,7),latent(size_t(phys)*pt*L,0xa5),rope(size_t(phys)*pt*R,0xa5),index(size_t(phys)*pt*(dim+4),0xa5),views(rows){
  for(int r=0;r<w;r++){map[r]=r;for(int p=0;p<P;p++)tables[r*P+p]=r*P+(P-1-p);auto&v=views[r];v.latent=latent.data();v.rotary=rope.data();v.index=index.data();v.index_dim=D;v.page_table=tables.data()+r*P;v.page_epochs=epochs.data();v.lease_epoch=7;v.page_tokens=pt;v.table_pages=P;v.physical_pages=phys;v.logical_length=T;v.reserved_tokens=P*pt;v.latent_dim=L;v.rotary_dim=R;}
 }
 int physical(int r,int key)const{return tables[r*P+key/pt]*pt+key%pt;}
 void clean(){for(int e:err)ck(e==0,"page error");}
};
static void cache(Module&m,int t,int w,int width,int component){
 Fixture f(t,w,128,width,64);if(component==1){f.R=width;f.rope.assign(size_t(f.phys)*f.pt*width,0xa5);for(auto&v:f.views){v.rotary=f.rope.data();v.rotary_dim=width;}}
 std::vector<float>x(w*width,1.f);auto*xp=x.data();auto*vp=f.views.data();auto*mp=f.map.data();auto*pp=f.pos.data();auto*ep=f.err.data();int blocks=(w*width+127)/128;
 void*args[]={&xp,&vp,&mp,&pp,&ep,&w,&width,&component};
 {Region r(component?"paged_cache_component_store.rotary":"paged_cache_component_store.latent",{{"T",t},{"w",w},{"width",width},{"elements",w*width},{"pages",f.P},{"blocks",blocks}});m.launch("paged_cache_component_store",dim3(blocks),dim3(128),args);}
 auto&pool=component?f.rope:f.latent;std::vector<uint8_t>expected(pool.size(),0xa5);for(int r=0;r<w;r++)std::fill_n(expected.begin()+size_t(f.physical(r,t-1))*width,width,0x38);ck(pool==expected,"component numeric/history guard");f.clean();
}
static void append(Module&m,int t,int w,int ld,int rd){
 Fixture f(t,w,128,ld,rd);int batches=w,rows=1;std::vector<int>offset(w,t-1);std::vector<float>x(w*ld,1.f),y(w*rd,-1.f);auto*vp=f.views.data();auto*op=offset.data();auto*xp=x.data();auto*yp=y.data();auto*ep=f.err.data();int blocks=(w*(ld+rd)+127)/128;
 void*args[]={&vp,&op,&xp,&yp,&ep,&batches,&rows,&ld,&rd};
 {Region r("paged_fp8_append",{{"T",t},{"w",w},{"latent",ld},{"rotary",rd},{"elements",w*(ld+rd)},{"pages",f.P},{"blocks",blocks}});m.launch("paged_fp8_append",dim3(blocks),dim3(128),args);}
 std::vector<uint8_t>ex(f.latent.size(),0xa5),ey(f.rope.size(),0xa5);for(int r=0;r<w;r++){std::fill_n(ex.begin()+size_t(f.physical(r,t-1))*ld,ld,0x38);std::fill_n(ey.begin()+size_t(f.physical(r,t-1))*rd,rd,0xb8);}ck(ex==f.latent&&ey==f.rope,"append oracle/history guards");f.clean();
}
static void prepare_k(Module&m,int t,int w){
 Fixture f(t,w);std::vector<float>x(w*128),weight(128,0),bias(128,1),freq(32,0);for(int i=0;i<w*128;i++)x[i]=float(i%7)-3;
 auto*xp=x.data();auto*wp=weight.data();auto*bp=bias.data();auto*fp=freq.data();auto*vp=f.views.data();auto*mp=f.map.data();auto*pp=f.pos.data();auto*ep=f.err.data();float eps=1e-5;
 void*args[]={&xp,&wp,&bp,&fp,&vp,&mp,&pp,&ep,&w,&eps};
 {Region r("indexer_prepare_k",{{"T",t},{"w",w},{"pages",f.P},{"heads",32},{"dim",128},{"blocks",w}});m.launch("prepare_k_kernel",dim3(w),dim3(128),args);}
 auto ex=f.index;std::fill(ex.begin(),ex.end(),0xa5);for(int r=0;r<w;r++){int p=f.physical(r,t-1);std::fill_n(ex.begin()+size_t(p)*128,128,0x78);float val=1.f/256;std::copy_n(reinterpret_cast<uint8_t*>(&val),4,ex.begin()+size_t(f.phys)*256*128+4*p);}ck(ex==f.index,"prepare K value/scale/history guards");f.clean();
}
static void glue(Module&m,int t,int w,int k){
 Fixture f(t,w);auto*vp=f.views.data();auto*mp=f.map.data();auto*pp=f.pos.data();auto*ep=f.err.data();
 std::vector<int>ids(w*k+2,-999),slots(w*k+2,-999),ks(w+2,-999),ke(w+2,-999);auto*ip=ids.data()+1;auto*sp=slots.data()+1;auto*kp=ks.data()+1;auto*ekp=ke.data()+1;int blocks=(w*k+255)/256;
 {void*args[]={&pp,&w,&k,&ip};Region r("indexer_fill_dense",{{"T",t},{"w",w},{"K",k},{"wK",w*k},{"blocks",blocks}});m.launch("fill_dense_kernel",dim3(blocks),dim3(256),args);}
 for(int r=0;r<w;r++)for(int j=0;j<k;j++)ck(ip[r*k+j]==(j<t?j:-1),"dense ids oracle");ck(ids.front()==-999&&ids.back()==-999,"dense guard");
 for(int r=0;r<w;r++)for(int j=0;j<k;j++)ip[r*k+j]=(j%5==0)?-1:int(int64_t(j)*(t-1)/std::max(1,k-1));
 {void*args[]={&ip,&vp,&mp,&w,&k,&sp};Region r("indexer_ids_to_slots",{{"T",t},{"w",w},{"K",k},{"wK",w*k},{"valid_ids",w*(k-(k+4)/5)},{"pages",f.P},{"blocks",blocks}});m.launch("ids_to_slots_kernel",dim3(blocks),dim3(256),args);}
 for(int r=0;r<w;r++)for(int j=0;j<k;j++)ck(sp[r*k+j]==(ip[r*k+j]<0?-1:f.physical(r,ip[r*k+j])),"slots oracle");ck(slots.front()==-999&&slots.back()==-999,"slot guards");
 GatherPlan plan;plan.requests=w;for(int r=0;r<w;r++){plan.request[r]=r;plan.offset[r]=r*t;plan.count[r]=t;plan.base_of_view[r]=r*t;}
 {void*args[]={&mp,&pp,&plan,&w,&kp,&ekp};Region r("indexer_row_ranges",{{"T",t},{"w",w},{"K",k},{"blocks",1}});m.launch("row_ranges_kernel",dim3(1),dim3(128),args);}
 for(int r=0;r<w;r++)ck(kp[r]==r*t&&ekp[r]==(r+1)*t,"range oracle");ck(ks.front()==-999&&ks.back()==-999&&ke.front()==-999&&ke.back()==-999,"range guards");f.clean();
}
static void gather(Module&m,int t,int w){
 Fixture f(t,w);GatherPlan plan;plan.requests=w;int total=t*w,padded=(total+3)/4*4;for(int r=0;r<w;r++){plan.request[r]=r;plan.offset[r]=r*t;plan.count[r]=t;plan.base_of_view[r]=r*t;}
 auto*scale=reinterpret_cast<float*>(f.index.data()+size_t(f.phys)*256*128);for(int r=0;r<w;r++)for(int key=0;key<t;key++){int p=f.physical(r,key);for(int d=0;d<128;d++)f.index[size_t(p)*128+d]=uint8_t((r*19+key+d)%125);scale[p]=float(key+1);}
 std::vector<uint8_t>out(size_t(total)*128+32,0xe7);std::vector<float>sc(padded+2,-999);auto*op=out.data()+16;auto*sp=sc.data()+1;auto*vp=f.views.data();auto*ep=f.err.data();int blocks=(padded*8+255)/256;
 void*args[]={&vp,&plan,&op,&sp,&ep,&total,&padded};
 {Region r("indexer_gather_K.prefill_fallback",{{"T",t},{"w",w},{"tokens",total},{"padded_tokens",padded},{"request_scan_work",t*w*(w-1)/2},{"pages",f.P},{"blocks",blocks}});m.launch("gather_kernel",dim3(blocks),dim3(256),args);}
 for(int r=0;r<w;r++)for(int key=0;key<t;key++){for(int d=0;d<128;d++)ck(op[(size_t(r)*t+key)*128+d]==uint8_t((r*19+key+d)%125),"gather bytes");ck(sp[r*t+key]==key+1,"gather scales");}for(int j=total;j<padded;j++)ck(sp[j]==0,"scale padding");for(int i=0;i<16;i++)ck(out[i]==0xe7&&out[out.size()-1-i]==0xe7,"gather guards");ck(sc.front()==-999&&sc.back()==-999,"scale guards");f.clean();
}
static void validate(Module&m,int t,int w,int pt){
 Fixture f(t,w,128,128,64,pt);auto*vp=f.views.data();auto*ep=f.err.data();int dim=128,pages=(t+pt-1)/pt;
 void*args[]={&vp,&w,&dim,&ep};
 {Region r("paged_indexer_validate_pages",{{"T",t},{"w",w},{"page_tokens",pt},{"pages",pages},{"wPages",w*pages},{"blocks",w}});m.launch("paged_indexer_validate_pages",dim3(w),dim3(128),args);}
 f.clean();
}
static void scoring(Module&m,int t,int w,int h,int d,int tile,int k){
 Fixture f(t,w,d);for(int r=0;r<w;r++)for(int key=0;key<t;key++)std::fill_n(f.index.begin()+size_t(f.physical(r,key))*d,d,0x38);
 std::vector<float>q(size_t(w)*h*d,1.f),weights(w*h,1.f),scores(w*t+2,-999);auto*qp=q.data();auto*wp=weights.data();auto*sp=scores.data()+1;auto*vp=f.views.data();auto*mp=f.map.data();auto*pp=f.pos.data();auto*ep=f.err.data();int tiles=(t+tile-1)/tile,blocks=w*tiles;
 void*args[]={&qp,&wp,&sp,&vp,&mp,&pp,&ep,&w,&t,&h,&d,&tile};
 {Region r("paged_indexer_scores_tiled.fallback",{{"T",t},{"w",w},{"K",k},{"heads",h},{"dim",d},{"pages",f.P},{"blocks",blocks},{"wT",w*t},{"wTH",w*t*h},{"wTHD",w*t*h*d},{"tileHD",blocks*h*d},{"tileH",blocks*h}});m.launch("paged_indexer_scores_tiled",dim3(blocks),dim3(128),args,size_t(d)*33*sizeof(float));}
 for(int i=0;i<w*t;i++)ck(std::abs(sp[i]-std::sqrt(float(h*d)))<2e-4f*std::max(1.f,std::sqrt(float(h*d))),"scores uniform oracle");ck(scores.front()==-999&&scores.back()==-999,"score guards");f.clean();
}
static void native(Module&m,int t,int w,int d,bool unit){int n=w*d,offset=t*d,blocks=(n+127)/128;float scale=1;std::vector<float>x(n,1.f);std::vector<uint8_t>y(size_t(offset)+n+2,0xe7);auto*xp=x.data();auto*yp=y.data()+1;void*args[]={&xp,&yp,&n,&offset,&scale};
 {Region r(unit?"cache_fp8_store_unit.legacy":"cache_fp8_store_native.legacy",{{"T",t},{"w",w},{"dim",d},{"elements",n},{"blocks",blocks}});m.launch(unit?"cache_fp8_store_unit":"cache_fp8_store_native",dim3(blocks),dim3(128),args);}
 for(int i=0;i<int(y.size());i++)ck(y[i]==(i>=offset+1&&i<offset+n+1?0x38:0xe7),"legacy store guards");}
static void residual(Module&m,int t,int w,int hidden,bool bf16){int n=w*hidden,blocks=(n+127)/128;std::vector<float>x(n,1.f),sum(n,2.f),out(n+2,-999);std::vector<uint16_t>sum16(n,0x4000);auto*xp=x.data();void*sp=bf16?static_cast<void*>(sum16.data()):static_cast<void*>(sum.data());auto*op=out.data()+1;void*args[]={&xp,&sp,&op,&n};
 {Region r(bf16?"lib_ar_residual_bf16.unfused":"lib_ar_residual_f32.unfused",{{"T",t},{"w",w},{"hidden",hidden},{"elements",n},{"blocks",blocks}});m.launch(bf16?"lib_ar_residual_bf16":"lib_ar_residual_f32",dim3(blocks),dim3(128),args);}
 for(int i=0;i<n;i++)ck(op[i]==3.f,"residual oracle");ck(out.front()==-999&&out.back()==-999,"residual guards");}
static void topk(Module&m,int t,int w,int k,int pattern,bool hist){
 std::vector<float>scores(w*t);std::vector<int>length(w,t),out(w*k+2,-999);for(int r=0;r<w;r++)for(int i=0;i<t;i++)scores[r*t+i]=pattern==0?float(i+1):pattern==1?1.f:float((i*73)%t+1);auto*sp=scores.data();auto*lp=length.data();auto*op=out.data()+1;
 std::string name=hist?"topk_histogram.smallK.":"topk_fallback.";name+=pattern==0?"ascending":pattern==1?"ties":"permuted";
 std::vector<int> bins(hist?4096:2048);for(int i=0;i<t;i++){uint32_t bits;std::memcpy(&bits,&scores[i],4);unsigned half=(bits>>13)-((127-15)<<10);int bin=hist?((half|0x8000)>>4):((~half&0x7fff)>>5);bins[bin]++;}
 int M=0,above=0,bin=0;if(hist){for(bin=4095;bin>=0;--bin){if(above+bins[bin]>=k){M=bins[bin];break;}above+=bins[bin];}}else{for(bin=0;bin<2048;++bin){if(above+bins[bin]>=k){M=bins[bin];break;}above+=bins[bin];}}
 int ties=std::min(M,k),ballot=(hist&&above+M>k&&ties<=64)?ties*ties:0;
 {Region r(name.c_str(),{{"T",t},{"w",w},{"K",k},{"wT",w*t},{"wK",w*k},{"blocks",w},{"heads",32},{"wCandidates",w*M},{"wCandidateSquared",int64_t(w)*M*M},{"wAbove",w*above},{"wScanRounds",w*(hist?1:(bin/512+1))},{"wBallotPairs",w*ballot},{"wBallotSmallPairs",ties<=32?w*ballot:0},{"wBallotMediumPairs",ties>32?w*ballot:0}});if(hist){void*args[]={&sp,&op,&t,&t};m.launch(("topk_hist"+std::to_string(k)).c_str(),dim3(w),dim3(1024),args,40000);}else{void*args[]={&sp,&lp,&op,&t,&k};m.launch("topk_fallback",dim3(w),dim3(512),args,k*4);}}
 for(int r=0;r<w;r++){std::vector<int>seen(t);for(int j=0;j<k;j++){int id=op[r*k+j];ck(id>=0&&id<t&&!seen[id],"topk valid unique");seen[id]=1;if(pattern!=1)ck(scores[r*t+id]>=float(t-k+1),"topk selected set");}}
 ck(out.front()==-999&&out.back()==-999,"topk guards");
}
int main(int argc,char**argv){try{ck(argc>=3,"usage: driver module mode");Module m(argv[1]);std::string mode=argv[2];
 if(mode=="glue")for(int t:{32,64,128,256,512})for(int w:{1,2,3,4}){glue(m,t,w,8+8*(w%3));gather(m,t,w);prepare_k(m,t,w);validate(m,t,w,256);validate(m,t,w,32);std::cout<<"glue T="<<t<<" w="<<w<<" PASS\n";}
 if(mode=="cache")for(int t:{32,64,128,256,512})for(int w:{1,2,3,4}){for(int d:{32,128,256}){cache(m,t,w,d,0);cache(m,t,w,d,1);native(m,t,w,d,false);native(m,t,w,d,true);}append(m,t,w,128,64);append(m,t,w,256,32);std::cout<<"cache T="<<t<<" w="<<w<<" PASS\n";}
 if(mode=="scoring"){for(int t:{32,64,128,256,512})for(int w:{1,2,3,4})for(int h:{1,4})scoring(m,t,w,h,16,128,8+8*(w%3));for(int h:{2,8,16,32})for(int d:{8,32,64,128})scoring(m,32,1,h,d,128,16);for(int t:{33,65,129})scoring(m,t,2,2,32,64,8);std::cout<<"scoring PASS\n";}
 if(mode=="scoring_refined"){for(int t:{32,64,128})for(int w:{1,2})for(int tile:{16,32})scoring(m,t,w,32,128,tile,16);for(int t:{256,512})scoring(m,t,1,4,32,16,8);std::cout<<"scoring actual tile sizes PASS\n";}
 if(mode=="residual")for(int t:{32,64,128,256,512})for(int w:{1,2,3,4})for(int hidden:{32,128,256,512})for(bool bf16:{false,true})residual(m,t,w,hidden,bf16);
 if(mode=="topk")for(bool hist:{false,true})for(int pattern:{0,1,2})for(int t:{32,64,128,256,512})for(int w:{1,2,3,4}){int k=w==1?8:w==2?16:w==3?8:16;topk(m,t,w,k,pattern,hist);std::cout<<"topk hist="<<hist<<" pattern="<<pattern<<" T="<<t<<" w="<<w<<" PASS"<<std::endl;}
 if(mode=="glue_refined")for(int t:{32,65,128,255,512})for(int w:{1,2,3,4}){for(int k:{8,16,24,64})glue(m,t,w,k);gather(m,t,w);prepare_k(m,t,w);validate(m,t,w,256);validate(m,t,w,32);std::cout<<"glue refined T="<<t<<" w="<<w<<" PASS\n";}
 if(mode=="topk_refined")for(bool hist:{false,true})for(int pattern:{0,1,2})for(int t:{32,64,128,256,512})for(int w:{1,2,3,4})for(int k:{8,16,32,64})if(k<t){topk(m,t,w,k,pattern,hist);std::cout<<"topk refined hist="<<hist<<" pattern="<<pattern<<" T="<<t<<" w="<<w<<" K="<<k<<" PASS"<<std::endl;}
 if(mode=="hist_ties")for(int t:{32,64,128,256,512})for(int w:{1,2,3,4})for(int k:{8,16,32,64})if(k<t)topk(m,t,w,k,1,true);
 if(mode=="geometry")for(int w:{1,2,3,4})for(int k:{96,128})glue(m,65,w,k);
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
