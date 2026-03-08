# Stochastic Discrete Event-Modelling-and-Water-Meter
This is the official code repository for the paper _"Unlocking Household Water End-Use Patterns from Low Time-Resolution Smart Meter Data via Stochastic Discrete Event Modeling and Bayesian Inference"_ under review. 

Sample data is provided to apply Approximate Bayesian Computation (ABC) to calibrate an stochastic household water end-use model (SThWT) using only hourly water meter data

## 📄 Third-party Code and License Notice
This project includes and modifies source code from the open-source project **[pysimdeum](https://github.com/KWR-Water/pysimdeum)**, which is licensed under the **European Union Public Licence v.1.2 (EUPL-1.2)**.

### 🔗 Referenced Files from `pysimdeum`:

The following files were originally part of `pysimdeum` and have been modified in this project:

- `pysimdeum/core/house.py`  
- `pysimdeum/core/end_use.py`  
- `pysimdeum/core/statistics.py`  

## ABC settings 
### Seeds: DEFAULT (system time-based)
   pyABC uses system timestamp as seed by default
### Tolerances: 
Adaptive P-Norm Distance with MAD scaling
distance = AdaptivePNormDistance(
     p=1,                              # L1 norm (Manhattan distance)
    scale_function=mad,    # Median Absolute Deviation for normalization
)
 - Initial epsilon: Before the first generation, ABC randomly sampled a set of parameters from the prior distributions and generated simulations. The initial MADs are then calculated across all these simulations for each summary statistic. 
 - Reduction strategy: Adaptive quantile-based (default: median)
 - No minimum epsilon specified
 - Terminates by max_nr_populations or max_walltime
### Transition Kernel: DEFAULT

