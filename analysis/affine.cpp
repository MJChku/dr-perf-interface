// Offline affine checker. No DynamoRIO or Python runtime dependencies.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace {
struct Elimination { size_t row, pivot; double factor; };
struct Plan {
    size_t n, k;
    std::vector<double> values, mean, scale, x, matrix;
    std::vector<int> pivot;
    std::vector<Elimination> operations;
    Plan(size_t rows, size_t columns, const double *v, const double *m, const double *s)
        : n(rows), k(columns), values(v, v + n*k), mean(m, m+k), scale(s, s+k),
          x(n*k), matrix(k*k, 0.0), pivot(k, -1) {
        for (size_t i=0; i<n; ++i)
            for (size_t j=0; j<k; ++j) x[i*k+j]=(values[i*k+j]-mean[j])/scale[j];
        double magnitude=1.0;
        for (size_t p=0; p<k; ++p) for (size_t q=0; q<k; ++q) {
            double total=0.0;
            for (size_t i=0; i<n; ++i) total+=x[i*k+p]*x[i*k+q];
            matrix[p*k+q]=total;
            magnitude=std::max(magnitude,std::abs(total));
        }
        std::vector<bool> used(k,false);
        for (size_t c=0; c<k; ++c) {
            size_t p=k;
            for (size_t r=0; r<k; ++r)
                if (!used[r] && (p==k || std::abs(matrix[r*k+c])>std::abs(matrix[p*k+c]))) p=r;
            if (p==k || std::abs(matrix[p*k+c])<=1e-9*magnitude) continue;
            used[p]=true; pivot[c]=static_cast<int>(p);
            for (size_t r=0; r<k; ++r) if (r!=p && matrix[r*k+c]!=0.0) {
                double f=matrix[r*k+c]/matrix[p*k+c];
                operations.push_back({r,p,f});
                for (size_t j=0; j<k; ++j) matrix[r*k+j]-=f*matrix[p*k+j];
            }
        }
    }
    // Output: k coefficients, intercept, max deviation, mean, maximum, accepted.
    void fit(const double *y, double abs_tol, double rel_tol, double *out) const {
        double my=0.0, ymax=y[0];
        for (size_t i=0; i<n; ++i) { my+=y[i]; ymax=std::max(ymax,y[i]); }
        my/=static_cast<double>(n);
        std::vector<double> rhs(k,0.0);
        for (size_t p=0; p<k; ++p)
            for (size_t i=0; i<n; ++i) rhs[p]+=x[i*k+p]*(y[i]-my);
        for (const auto &op:operations) rhs[op.row]-=op.factor*rhs[op.pivot];
        double offset=0.0;
        for (size_t j=0; j<k; ++j) {
            out[j]=pivot[j]<0 ? 0.0 : rhs[pivot[j]]/matrix[pivot[j]*k+j]/scale[j];
            offset+=out[j]*mean[j];
        }
        const double constant=my-offset;
        double deviation=0.0;
        bool accepted=std::isfinite(constant);
        for (size_t i=0; i<n; ++i) {
            double prediction=0.0;
            for (size_t j=0; j<k; ++j) prediction+=out[j]*values[i*k+j];
            const double error=std::abs(y[i]-(prediction+constant));
            deviation=std::max(deviation,error);
            if (!std::isfinite(error) || error>std::max(abs_tol,rel_tol*y[i])) accepted=false;
        }
        out[k]=constant; out[k+1]=deviation; out[k+2]=my; out[k+3]=ymax;
        out[k+4]=accepted ? 1.0 : 0.0;
    }
};
}
extern "C" {
unsigned drperf_fit_abi() { return 1; }
void *drperf_fit_create(size_t n, size_t k, const double *v, const double *m, const double *s) {
    if (!n || !v || !m || !s) return nullptr;
    try { return new Plan(n,k,v,m,s); } catch (...) { return nullptr; }
}
void drperf_fit_destroy(void *p) { delete static_cast<Plan *>(p); }
void drperf_fit_dependent(void *p, unsigned char *out) {
    auto &plan=*static_cast<Plan *>(p);
    for (size_t j=0; j<plan.k; ++j) out[j]=plan.pivot[j]<0;
}
int drperf_fit_batch(void *p, size_t count, const double *ys, double abs_tol,
                     double rel_tol, double *out) {
    try {
        auto &plan=*static_cast<Plan *>(p);
        for (size_t b=0; b<count; ++b)
            plan.fit(ys+b*plan.n,abs_tol,rel_tol,out+b*(plan.k+5));
        return 0;
    } catch (...) { return -1; }
}
}
