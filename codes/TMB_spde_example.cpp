// include libraries
#include <TMB.hpp>
#include <Eigen/Sparse>
#include <vector>
#include <string>

using namespace density;
using Eigen::SparseMatrix;

// dcauchy for hyperparameters
template<class Type>
Type dcauchy(Type x, Type mean, Type shape, int give_log=0){
  Type logres = 0.0;
  logres-= log(M_PI);
  logres-= log(shape);
  // Note, this is unstable and should switch to log1p formulation
  logres-= log(1 + pow( (x-mean)/shape ,2));
  if(give_log) return logres; else return exp(logres);
}


// helper function to make sparse SPDE precision matrix
// Inputs:
// logkappa: log(kappa) parameter value
// logtau: log(tau) parameter value
//  M0, M1, M2: these sparse matrices are output from R::INLA::inla.spde2.matern()$param.inla$M*
template<class Type>
  SparseMatrix<Type> spde_Q(Type logkappa, Type logtau, SparseMatrix<Type> M0, SparseMatrix<Type> M1, SparseMatrix<Type> M2) {
  SparseMatrix<Type> Q;
  Type kappa2 = exp(2. * logkappa);
  Type kappa4 = kappa2*kappa2;
  Q = pow(exp(logtau), 2.)  * (kappa4*M0 + Type(2.0)*kappa2*M1 + M2);
  return Q;
}


template<class Type>
Type objective_function<Type>::operator() ()
{


//=============================================================================================================
//                                              DATA SECTION
//=============================================================================================================
  
// Vectors of real data
   DATA_VECTOR(y_i);         // y_i (response variable)
  
// Indices for factors
   DATA_FACTOR(site_i);       // Random effect index for observation i
  
  
// SPDE objects
   DATA_SPARSE_MATRIX(M0);    // used to make gmrf precision
   DATA_SPARSE_MATRIX(M1);    // used to make gmrf precision
   DATA_SPARSE_MATRIX(M2);    // used to make gmrf precision
  
  
//=============================================================================================================
//                                              PARAMETERS SECTION
//=============================================================================================================
  
// Fixed effects
  PARAMETER(logsigma);		         
  PARAMETER(logtau);		                 // spatial process
  PARAMETER(logkappa);		               // decorrelation distance (kind of)
  
  PARAMETER_VECTOR(omega_s);	           // spatial effects
  SparseMatrix<Type> Q   = spde_Q(logkappa, logtau, M0, M1, M2);
  

//===================================
//               Priors
//===================================
//    Type nlp=0.0;                          // negative log prior  (priors)
// 
// // Variance component
      Type sigma = exp(logsigma);
//    nlp -= dcauchy(sigma,   Type(0.0), Type(2.0));
//   
//   
// // Hyperpriors
//    Type tau   = exp(logtau);
//    Type kappa = exp(logkappa);
//    
//    nlp -= dnorm(tau,    Type(0.0),   Type(1.0), true);
//    nlp -= dnorm(kappa,  Type(0.0),   Type(1.0), true);
//    
     
   
   
//=============================================================================================================
// Objective function is sum of negative log likelihood components
   using namespace density;
   int n_i = y_i.size();	             // number of observations 
   Type nll_omega=0;		               // spatial effects
  
   
// The model predicted for each observation, in natural space:
   vector<Type> mu(n_i);
   for( int i=0; i<n_i; i++){
        mu(i) = omega_s(site_i(i)); 
}

// Probability of random effects
   //nll_omega += SCALE(GMRF(Q), 1/exp(logtau) )(omega_s);
     nll_omega += GMRF(Q)(omega_s); // returns negative already
 
// Probability of the data, given random effects (likelihood)
   vector<Type> log_lik(n_i);
   for( int i = 0; i<n_i; i++){
       log_lik(i) = dnorm(y_i(i), mu(i), sigma, true);
}
  
   Type nll = -log_lik.sum(); // total NLL
   
// Jacobian adjustment for variances
   //nll -= logsigma + logtau + logkappa;
   
// Calculate joint negative log likelihood
   //Type jnll = nll + nll_omega + nlp;
   Type jnll = nll + nll_omega;
  
   vector<Type> preds = mu;
    
    
// Derived quantities, given parameters
// Geospatial
   Type rho = sqrt(8) / exp(logkappa);
   Type sigma_s = 1 / sqrt(4 * M_PI * exp(2*logtau) * exp(2*logkappa));
   
  
//=====================================================================================================
// Reporting
   REPORT(jnll);
   REPORT(nll);
   REPORT(nll_omega);
  
    
   REPORT(omega_s);
   REPORT(preds);
   REPORT(logsigma);
   REPORT(logtau);
   REPORT(logkappa);
   REPORT(log_lik);
   REPORT(rho);		         
   REPORT(sigma_s);		         
  
//=======================================================================================================
// AD report (standard devations)
   ADREPORT(logsigma);	                  
   ADREPORT(logtau);
   ADREPORT(logkappa);
    
// Derived geospatial components
   ADREPORT(rho);		               // geostatistical range
   ADREPORT(sigma_s);		         
   return jnll;
}

