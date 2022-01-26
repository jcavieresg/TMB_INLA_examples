#include <TMB.hpp>

template<class Type>
  Type objective_function<Type>::operator() ()
{
  
  // DATA SECTION
  DATA_VECTOR(y); // Response variable (Gaussian distribution)
  DATA_VECTOR(x); // Contionious covariate
  
  
  // PARAMETER SECTION
  PARAMETER(beta0); // Intercept
  PARAMETER(beta1); // Parabeter realetd with x
  
  PARAMETER(logsigma); // logsigma
  
  Type sigma= exp(logsigma); //exp
  
  // We need to declare nll
  Type nll = 0;
  
  
  // Model
  vector<Type> mu = beta0 + x*beta1;
  nll -= sum(dnorm(y, mu, sigma, true));

  
  // Report section
  REPORT(beta0);
  REPORT(beta1);
  REPORT(sigma);
  
  ADREPORT(beta0);
  ADREPORT(beta1);
  ADREPORT(sigma);
  
  return(nll);
}
