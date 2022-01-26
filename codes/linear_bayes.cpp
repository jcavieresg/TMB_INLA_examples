#include <TMB.hpp>

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


template<class Type>
  Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(y);
  DATA_VECTOR(x);
  
  PARAMETER(beta0);
  PARAMETER(beta1);
  PARAMETER(logsigma);
  
  Type sigma= exp(logsigma);
  Type nll = 0;
  
  
  
  //===================================
  //               Priors
  //===================================
  Type nlp=0.0;                          // negative log prior  (priors)
  
// Prior on betas
  nlp-= dnorm(beta0,     Type(0.0), Type(1.0), true);
  nlp-= dnorm(beta1,     Type(0.0), Type(1.0), true);

  
// Prior on the variance 
  nlp -= dcauchy(sigma,   Type(0.0), Type(2.0));
  
  
  
// Mean of the model
  vector<Type> mu = beta0 + x*beta1;
  
// loglikelihood
  nll -= sum(dnorm(y, mu, sigma, true));
  
  
// Calculate joint negative log likelihood
  Type jnll = nll + nlp;
  
  
  //=====================================================================================================
  //  Reporting
  REPORT(nll);
  REPORT(nlp);
  REPORT(jnll);
  
  REPORT(beta0);
  REPORT(beta1);
  REPORT(logsigma);
  
  
  
  ADREPORT(beta0);		            
  ADREPORT(beta1);	              
  ADREPORT(logsigma);	            

  return(jnll);
}
