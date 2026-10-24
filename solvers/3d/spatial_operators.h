// Geometry helpers for the node-based Green--Gauss and corrected face gradient.
#ifndef QUICK_CFD_LOCAL_SPATIAL_OPERATORS_H
#define QUICK_CFD_LOCAL_SPATIAL_OPERATORS_H
#include <algorithm>
#include <cmath>
#include <cstddef>

#ifdef __CUDACC__
#define QCFD_HD __host__ __device__
#else
#define QCFD_HD
#endif

namespace qcfd {

QCFD_HD inline double dot3(const double* a,const double* b){
 return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

QCFD_HD inline void corrected_face_gradient(
 double* gradient,const double* displacement,double value_difference){
 const double distance2=dot3(displacement,displacement);
 const double correction=(value_difference-dot3(gradient,displacement))/distance2;
 for(int j=0;j<3;++j)gradient[j]+=correction*displacement[j];
}

QCFD_HD inline bool mitchell_face_gradient(const double* center_direction,
                                            const double* edge1,const double* edge2,
                                            double center_difference,double edge1_difference,
                                            double edge2_difference,double* gradient){
 double c12[3]={edge1[1]*edge2[2]-edge1[2]*edge2[1],edge1[2]*edge2[0]-edge1[0]*edge2[2],edge1[0]*edge2[1]-edge1[1]*edge2[0]};
 double c20[3]={edge2[1]*center_direction[2]-edge2[2]*center_direction[1],edge2[2]*center_direction[0]-edge2[0]*center_direction[2],edge2[0]*center_direction[1]-edge2[1]*center_direction[0]};
 double c01[3]={center_direction[1]*edge1[2]-center_direction[2]*edge1[1],center_direction[2]*edge1[0]-center_direction[0]*edge1[2],center_direction[0]*edge1[1]-center_direction[1]*edge1[0]};
 const double determinant=dot3(center_direction,c12),scale=sqrt(dot3(center_direction,center_direction)*dot3(edge1,edge1)*dot3(edge2,edge2));
 if(!(fabs(determinant)>1e-13*scale))return false;
 for(int j=0;j<3;++j)gradient[j]=(center_difference*c12[j]+edge1_difference*c20[j]+edge2_difference*c01[j])/determinant;
 return true;
}

inline void cross3(const double* a,const double* b,double* result){
 result[0]=a[1]*b[2]-a[2]*b[1];
 result[1]=a[2]*b[0]-a[0]*b[2];
 result[2]=a[0]*b[1]-a[1]*b[0];
}

// Gradients of the four barycentric shape functions. `points` is a contiguous
// [4][3] array. The signed determinant is retained, so either tetrahedron
// orientation is accepted.
inline bool tetra_shape_gradients(const double* points,double* gradients){
 double e1[3],e2[3],e3[3],c23[3],c31[3],c12[3];
 for(int j=0;j<3;++j){e1[j]=points[3+j]-points[j];e2[j]=points[6+j]-points[j];e3[j]=points[9+j]-points[j];}
 cross3(e2,e3,c23);cross3(e3,e1,c31);cross3(e1,e2,c12);
 const double determinant=dot3(e1,c23);
 double scale=0;for(int j=0;j<3;++j)scale=std::max(scale,std::max(std::fabs(e1[j]),std::max(std::fabs(e2[j]),std::fabs(e3[j]))));
 if(!(std::fabs(determinant)>1e-14*scale*scale*scale)||!std::isfinite(determinant))return false;
 for(int j=0;j<3;++j){
  gradients[3+j]=c23[j]/determinant;
  gradients[6+j]=c31[j]/determinant;
  gradients[9+j]=c12[j]/determinant;
  gradients[j]=-gradients[3+j]-gradients[6+j]-gradients[9+j];
 }
 return true;
}

inline bool solve_small(double matrix[3][3],double rhs[3],double solution[3],int dimension){
 double augmented[3][4];double largest=0;
 for(int i=0;i<dimension;++i){for(int j=0;j<dimension;++j){augmented[i][j]=matrix[i][j];largest=std::max(largest,std::fabs(matrix[i][j]));}augmented[i][dimension]=rhs[i];}
 if(!(largest>0))return false;
 for(int column=0;column<dimension;++column){
  int pivot=column;for(int row=column+1;row<dimension;++row)if(std::fabs(augmented[row][column])>std::fabs(augmented[pivot][column]))pivot=row;
  if(std::fabs(augmented[pivot][column])<=1e-13*largest)return false;
  if(pivot!=column)for(int j=column;j<=dimension;++j)std::swap(augmented[pivot][j],augmented[column][j]);
  const double inverse=1/augmented[column][column];for(int j=column;j<=dimension;++j)augmented[column][j]*=inverse;
  for(int row=0;row<dimension;++row)if(row!=column){const double factor=augmented[row][column];for(int j=column;j<=dimension;++j)augmented[row][j]-=factor*augmented[column][j];}
 }
 for(int i=0;i<dimension;++i)solution[i]=augmented[i][dimension];return true;
}

// Bounded pseudo-Laplacian interpolation from surrounding cell centers to a
// mesh node. Unbounded weights reproduce affine fields. Bounding the raw
// pseudo-Laplacian factors to [0,2] is a deliberate robustness tradeoff used
// only where extrapolation would otherwise produce a negative or excessive
// nodal weight.
inline bool pseudo_laplacian_weights(const double* centers,int count,const double* node,
                                     double* weights,bool& bounded,bool limit_weights=true,
                                     double epsilon=0,int maximum_dimension=3,int* dimension_out=nullptr){
 bounded=false;if(count<1)return false;maximum_dimension=std::max(0,std::min(3,maximum_dimension));
 double basis[3][3]={},origin[3]={centers[0],centers[1],centers[2]};int dimension=0,farthest=0;double farthest2=0;
 for(int i=1;i<count;++i){double d[3];for(int j=0;j<3;++j)d[j]=centers[3*i+j]-origin[j];const double distance2=dot3(d,d);if(distance2>farthest2){farthest2=distance2;farthest=i;}}
 if(maximum_dimension>0&&farthest2>epsilon*epsilon){dimension=1;const double inverse=1/std::sqrt(farthest2);for(int j=0;j<3;++j)basis[0][j]=(centers[3*farthest+j]-origin[j])*inverse;}
 int off_line=0;double line_distance2=0;
 if(dimension==1&&maximum_dimension>1)for(int i=0;i<count;++i){double d[3],projection=0;for(int j=0;j<3;++j){d[j]=centers[3*i+j]-origin[j];projection+=d[j]*basis[0][j];}double residual[3];for(int j=0;j<3;++j)residual[j]=d[j]-projection*basis[0][j];const double distance2=dot3(residual,residual);if(distance2>line_distance2){line_distance2=distance2;off_line=i;}}
 if(dimension==1&&maximum_dimension>1&&line_distance2>epsilon*epsilon){dimension=2;double d[3],projection=0;for(int j=0;j<3;++j){d[j]=centers[3*off_line+j]-origin[j];projection+=d[j]*basis[0][j];}const double inverse=1/std::sqrt(line_distance2);for(int j=0;j<3;++j)basis[1][j]=(d[j]-projection*basis[0][j])*inverse;}
 if(dimension==2&&maximum_dimension>2){cross3(basis[0],basis[1],basis[2]);double plane_distance=0;for(int i=0;i<count;++i){double d[3];for(int j=0;j<3;++j)d[j]=centers[3*i+j]-origin[j];plane_distance=std::max(plane_distance,std::fabs(dot3(d,basis[2])));}if(plane_distance>epsilon)dimension=3;}
 double scale[3]={1,1,1},moment[3]={},matrix[3][3]={};
 for(int a=0;a<dimension;++a){scale[a]=0;for(int i=0;i<count;++i){double d[3];for(int j=0;j<3;++j)d[j]=centers[3*i+j]-node[j];scale[a]=std::max(scale[a],std::fabs(dot3(d,basis[a])));}if(!(scale[a]>0))dimension=a;}
 for(int i=0;i<count;++i){double physical[3];for(int j=0;j<3;++j)physical[j]=centers[3*i+j]-node[j];double d[3]={};for(int a=0;a<dimension;++a){d[a]=dot3(physical,basis[a])/scale[a];moment[a]+=d[a];}for(int a=0;a<dimension;++a)for(int b=0;b<dimension;++b)matrix[a][b]+=d[a]*d[b];}
 double lambda[3]={};while(dimension>0&&!solve_small(matrix,moment,lambda,dimension))--dimension;if(dimension_out)*dimension_out=dimension;double sum=0;
 for(int i=0;i<count;++i){double raw=1;
  double physical[3];for(int j=0;j<3;++j)physical[j]=centers[3*i+j]-node[j];for(int a=0;a<dimension;++a)raw-=lambda[a]*dot3(physical,basis[a])/scale[a];
  const double selected=limit_weights?std::max(0.,std::min(2.,raw)):raw;bounded=bounded||selected!=raw;weights[i]=selected;sum+=selected;
 }
 if(!(sum>0)||!std::isfinite(sum)){for(int i=0;i<count;++i)weights[i]=1./count;bounded=true;return false;}
 for(int i=0;i<count;++i)weights[i]/=sum;return true;
}

} // namespace qcfd

#undef QCFD_HD
#endif
