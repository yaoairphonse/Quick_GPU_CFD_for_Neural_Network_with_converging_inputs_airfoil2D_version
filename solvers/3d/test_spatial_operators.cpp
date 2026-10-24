#include "spatial_operators.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void close(double actual,double expected,double tolerance,const char* message){require(std::fabs(actual-expected)<=tolerance*std::max(1.,std::fabs(expected)),message);}

int main()try{
 const double points[12]={0,0,0, 2,0,0, .3,.02,0, .2,.01,.0004};double shape[12];require(qcfd::tetra_shape_gradients(points,shape),"anisotropic tetrahedron rejected");
 const double affine_gradient[3]={1.7,-.4,2.3};double recovered[3]={};
 for(int i=0;i<4;++i){const double value=.8+affine_gradient[0]*points[3*i]+affine_gradient[1]*points[3*i+1]+affine_gradient[2]*points[3*i+2];for(int j=0;j<3;++j)recovered[j]+=value*shape[3*i+j];}
 for(int j=0;j<3;++j)close(recovered[j],affine_gradient[j],2e-11,"tetrahedral affine gradient mismatch");

 const double node[3]={0,0,0};const double centers[18]={1,.02,.001, -1,.01,.002, .1,.8,.003, -.2,-.7,.002, .3,.2,.6, -.4,.1,-.5};double weights[6];bool bounded=false;
 require(qcfd::pseudo_laplacian_weights(centers,6,node,weights,bounded),"full-rank nodal stencil rejected");require(!bounded,"balanced nodal stencil unexpectedly bounded");
 double sum=0,moment[3]={};for(int i=0;i<6;++i){sum+=weights[i];require(weights[i]>=0&&weights[i]<=1,"invalid normalized nodal weight");for(int j=0;j<3;++j)moment[j]+=weights[i]*centers[3*i+j];}
 close(sum,1,2e-13,"nodal weights do not preserve constants");for(double value:moment)close(value,0,2e-13,"nodal weights do not reproduce affine fields");

 double face_gradient[3]={.4,-.2,.7};const double displacement[3]={.3,.1,-.2},difference=.8;qcfd::corrected_face_gradient(face_gradient,displacement,difference);
 close(qcfd::dot3(face_gradient,displacement),difference,2e-13,"corrected face gradient misses center difference");
 const double center_direction[3]={.2,-.1,.7},edge1[3]={1,.2,0},edge2[3]={-.1,.8,.3};double mitchell[3];
 require(qcfd::mitchell_face_gradient(center_direction,edge1,edge2,qcfd::dot3(affine_gradient,center_direction),qcfd::dot3(affine_gradient,edge1),qcfd::dot3(affine_gradient,edge2),mitchell),"Mitchell stencil rejected");
 for(int j=0;j<3;++j)close(mitchell[j],affine_gradient[j],2e-13,"Mitchell affine gradient mismatch");
 std::cout<<"spatial operator tests passed\n";return 0;
}catch(const std::exception& error){std::cerr<<"ERROR: "<<error.what()<<'\n';return 1;}
