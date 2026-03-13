# -*- coding: utf-8 -*-
"""
#The following code  is adapted the open-source project “ KWR-Water/pysimdeum”
#Original source:https://github.com/KWR-Water/pysimdeum.git
#License:European Union Public License 1.2

#Key modifications in this version by @uwaterloo-InfrasSUST:

#1.Determine the number of household members using pre-trial survey data.
#2.Determine available water use fixtures at home from pre-trial survey data.
#3.Aggregated 'bathroom tap' and 'kitchen tap' into a single 'kitchentap' class.(Original model treated them as separate end-uses; we combined them.The name 'kitchentap' is retained, but frequency = bathroomtap + kitchen tap.)
#4.Introduced predefined prior distributions for end-use intensity,duration and frequency,based on pre-trial survey data.
"""
import os
import tempfile
import pickle
from pyabc.distance import *
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from matplotlib import pyplot as plt
import random
from pysimdeum.core.statistics import Statistics
from pysimdeum.core.house import Property, HousePattern, House
from pysimdeum.core.user import User,Presence
from pysimdeum.core.end_use import EndUse, Bathtub, BathroomTap, Dishwasher, KitchenTap, OutsideTap, Shower, Wc, WashingMachine

import pysimdeum.core.end_use as EndUses
from pysimdeum.core.utils import chooser, duration_decorator, normalize, to_timedelta

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import xarray as xr
import copy
import pyabc
from pyabc import ABCSMC, RV, Distribution, LocalTransition, MedianEpsilon,QuantileEpsilon,MulticoreEvalParallelSampler
from pyabc.visualization import plot_data_callback, plot_kde_2d
import re
from datetime import timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
#read household profile 
households = pd.read_excel("/Users/fangxinyu/Desktop/Github/Household pre trial survey sample.xlsx")
#%% Washingmachine and Dishwasher pattern
# A complete washing machine cycle lasts 3,600 seconds, including 300 seconds for water filling
def washingmachine_enduse_pattern_revised(Ewm):

    value = Ewm

    index = pd.timedelta_range(start='00:00:00', freq='1s', periods=3600)
    s = pd.Series(0, index=index)
    s.iloc[0:120] = value
    s.iloc[1800:1980] = value

    # s.index = s.index - s.index[0]
    return s

# A complete dishwasher cycle lasts 3,600 seconds, including 180 seconds for water filling
def dishwasher_enduse_pattern_revised(Edw):

    value = Edw

    index = pd.timedelta_range(start='00:00:00', freq='1s', periods=3600)
    s = pd.Series(0, index=index)

    s.iloc[0:30] = value
    s.iloc[1800:1950] = value

    # s.index = s.index - s.index[0]

    return s

# from SIMDEUM
def usage_probability(time_resolution='1s'):
    """Produces uninformed prior.
    """
    # produce datetime index
    index = pd.timedelta_range(start='00:00:00', end='24:00:00', freq=time_resolution, closed='left')

    prob = pd.Series(data=1, index=index)  # ... uniform probability over time and cast it into pandas series.
    prob /= prob.sum()  # ... normalization of the probabilities

    return prob


def get_time_indices(enduse_time_records, enduse):
    return enduse_time_records.get(enduse, [])

#Note: Dishwashers and outdoor taps are not present in all households
#Appliance-specific simulation
# This function simulates water consumption only for households equipped with both dishwasher and outdoor tap
def simulate_water_consumption(pars):
            stats = Statistics()
            prop = Property(statistics=stats)
            
            # determine family type based on pre-trial survey
            house = prop.built_house(house_type=households.family_type[HHID])
            # job statistic
            job_stats = normalize(pd.Series(house.statistics.household[house.house_type]['job']))

            # division age statistics
            age_stats = normalize(pd.Series(house.statistics.household[house.house_type]['division_age']))

            # division gender statistics
            gender_stats = normalize(pd.Series(house.statistics.household[house.house_type]['division_gender']))

            if households.family_type[HHID] == 'family':
               # obtain number of household members from pre-trial survey
               num = households.household[HHID]
               job = chooser(job_stats)
               f_job, m_job = (job in ['both', 'only_female']), (job in ['both', 'only_male'])
               family = [User(id='user_1', gender='male', age='adult', job=f_job),  # father
                         User(id='user_2', gender='female', age='adult', job=m_job)]  # mother

               # add child/teen until family size is reached
               for numchild in range(2, num):
                   gender = chooser(gender_stats)
                   age = chooser(age_stats[['child', 'teen']])
                   family += [User(id='user_' + str(numchild+1), gender=gender, age=age, job=False)]  #
               house.users = family
            else:
               house.populate_house()

            # Identify available water fixtures in each household from survey responses
            for key,appliances in house.statistics.end_uses.items():
                classname = appliances['classname']
                u = np.random.uniform() * 100
                penetration = 100
                if classname in ['Wc', 'Shower']:
                   stype = chooser(appliances['subtype'], 'penetration')
                   eu_instance = getattr(EndUses, stype) (statistics=appliances)
                   house.appliances.append(eu_instance)
                elif classname in ['KitchenTap', 'WashingMachine','OutsideTap','Dishwasher']:
                   eu_instance = getattr(EndUses, classname)(statistics=appliances)
                   house.appliances.append(eu_instance)
                elif classname == 'Bathtub' and households.fre_bath_day[HHID] != 0 :
                   eu_instance = getattr(EndUses, classname)(statistics=appliances)
                   house.appliances.append(eu_instance)

            #create dataxarray for consumption
            date = datetime.now().date()
            timedelta = pd.to_timedelta('1 day')
            time = pd.date_range(start=date, end=date + timedelta, freq='1s')
            users = [x.id for x in house.users]+ ['household']
            enduse = [x.statistics['classname'] for x in house.appliances]
            patterns = [x for x in range(0, 1)]
            consumption = np.zeros((len(time), len(users), len(enduse), 1))
            num_patterns = 0
            numusers=len(house.users)
            
            #Derived prior probability of water use at each hour from meter data
            prob_usage = usage_probability().values
            prob_user = normalize(pdf).values

            simulation_records = {
                     'Tap': {'frequency': 0, 'durations': []},
                     'Wc': {'frequency': 0, 'durations': []},
                     'Shower': {'frequency': 0, 'durations': []},
                     'OutsideTap': {'frequency': 0, 'durations': []},
                     'WashingMachine': {'frequency': 0, 'durations': []},
                     'Dishwasher': {'frequency': 0, 'durations': []},
                     'Bathtub': {'frequency': 0, 'durations': []},
                     }


            for k, appliance in enumerate(house.appliances):
                # Kitchentap & Bathroomtap
                if appliance.__class__.__name__ == 'KitchenTap':
                    # gurantee freq > 0
                    total_freq = 0
                    #Bathroom taps are assumed to be used an average of 4.1 times per day by each household member
                    for j, user in enumerate(house.users):
                       freq =  np.random.poisson(4.1)
                       total_freq += freq
                       # mean tap duration=24.02s, standard deviation=1.3*mean
                       for i in range(freq):
                            #mean = np.log(24.02) - 0.5=2.678, sigma=1
                            duration = max(1, abs(int(np.random.lognormal(2.678,1))))
                            #record duration
                            simulation_records['Tap']['durations'].append(duration)
                            #intensity is sampled from prior distribution
                            intensity =  abs(pars['tap_intensity'])
                            #determine timing of event(SIMDEUM)
                            u = np.random.uniform()
                            prob_joint = normalize(prob_user * prob_usage)  
                            start = np.argmin(np.abs(np.cumsum(prob_joint) - u))
                            end = start + duration
                            consumption[start:end, j, k, 0] = intensity
                            
                    #Kitchen taps are used at household level
                    r, p = get_nb_parameters(len(house.users))
                    freq = np.random.negative_binomial(r, p)
                    total_freq += freq
                    j= len(house.users)
                    for i in range(freq):
                        duration = max(1, abs(int(np.random.lognormal(2.678,1))))
                        simulation_records['Tap']['durations'].append(duration)
                        #intensity is sampled from prior distribution
                        intensity =  abs(pars['tap_intensity'])
                        #determine timing of event(SIMDEUM)
                        u = np.random.uniform()
                        prob_joint = normalize(prob_user * prob_usage)  
                        start = np.argmin(np.abs(np.cumsum(prob_joint) - u))
                        end = start + duration
                        consumption[start:end, j, k, 0] = intensity
                    simulation_records['Tap']['frequency'] = total_freq


                # Toilet
                elif appliance.__class__.__name__ == 'WcNormalSave' or appliance.__class__.__name__ == 'WcNormal' or appliance.__class__.__name__ == 'WcNew' or appliance.__class__.__name__ == 'WcNewSave':
                     totalfreq =0
                     for j,user in enumerate(house.users):
                       # Obtain average daily per capita toilet frequency from pre-trial survey
                       freq = max(1, min(np.random.poisson(households.fre_flush_day[HHID]/(len(house.users))), 6))
                       totalfreq += freq
                       for i in range(freq):
                            duration = int(pd.Timedelta(minutes=3).total_seconds())
                            simulation_records['Wc']['durations'].append(duration)
                            intensity = abs(pars['Wc_intensity'])
                            prob_joint = normalize(prob_user * prob_usage)
                            u = np.random.uniform()
                            start = np.argmin(np.abs(np.cumsum(prob_joint) - u))
                            end = start + duration
                            consumption[start:end, j, k, 0] = intensity
                     simulation_records['Wc']['frequency'] = totalfreq
                     
                # Shower
                elif appliance.__class__.__name__ == 'NormalShower' or appliance.__class__.__name__ == 'FancyShower' :
                     j= len(house.users)
                     #Obtain average daily shower frequency per household from pre-trial survey
                     freq = max(1, min(np.random.poisson(households.fre_shower_day[HHID]), 2*j))
                     simulation_records['Shower']['frequency'] = freq
                     for i in range(freq):
                                m = np.random.chisquare(households.dur_shower[HHID])
                                duration = int(pd.Timedelta(minutes=abs(m)).total_seconds())
                                simulation_records['Shower']['durations'].append(duration)
                                intensity = abs(pars['shower_intensity'])
                                prob_joint = normalize(prob_user * prob_usage)
                                u = np.random.uniform()
                                start = np.argmin(np.abs(np.cumsum(prob_joint) - u))
                                end = start + duration
                                consumption[start:end, j, k, num_patterns] = intensity

                #Outsidetap
                elif appliance.__class__.__name__ == 'OutsideTap':
                        #Obtain average daily outside tap frequency per household from pre-trial survey
                        freq = np.random.poisson(households.fre_outdoor_day[HHID] )
                        simulation_records['OutsideTap']['frequency'] = freq
                        j= len(house.users)
                        #Obtain average daily outside tap duration per household from pre-trial survey
                        average = households.dur_outdoor_day[HHID]*60
                        sigma=1
                        mu = np.log(average) - 0.5 
                        for i in range(freq):
                            duration = max(1, abs(int(np.random.lognormal(mu,sigma))))
                            simulation_records['OutsideTap']['durations'].append(duration)
                            intensity = abs(pars['outsidetap_intensity'])
                            prob_joint = normalize(prob_user * prob_usage)
                            u = np.random.uniform()
                            start = np.argmin(np.abs(np.cumsum(prob_joint) - u))
                            end = start + duration
                            consumption[start:end, j, k, num_patterns] = intensity

                #Washing machine
                elif appliance.__class__.__name__ == 'WashingMachine':
                        # Obtain average daily washing machine frequency per household from pre-trial survey
                        freq = np.random.poisson(households.fre_washing_machine_day[HHID])
                        simulation_records['WashingMachine']['frequency'] = freq
                        prob_joint = normalize(prob_user * prob_usage)
                        pattern = washingmachine_enduse_pattern_revised(abs(pars['wm_intensity']))
                        j= len(house.users)
                        duration = len(pattern)
                        for i in range(freq):
                            simulation_records['WashingMachine']['durations'].append(duration)
                            u = np.random.random()
                            start = np.argmin(np.abs(np.cumsum(prob_joint) - u))
                            end = start + duration

                            if end > (24 * 60 * 60):
                                end = 24 * 60 * 60
                            difference = end - start
                            consumption[start:end, j, k, num_patterns] = pattern[:difference]

                #Dishwasher
                elif appliance.__class__.__name__ == 'Dishwasher':
                        freq = np.random.poisson(households.fre_dish_day[HHID])
                        simulation_records['Dishwasher']['frequency'] = freq
                        prob_joint = normalize(prob_user * prob_usage)
                        pattern = dishwasher_enduse_pattern_revised(abs(pars['dw_intensity']))
                        duration = len(pattern)
                        j= len(house.users)
                        for i in range(freq):
                            simulation_records['Dishwasher']['durations'].append(duration)
                            u = np.random.random()
                            start = np.argmin(np.abs(np.cumsum(prob_joint) - u))
                            end = start + duration

                            if end > (24 * 60 * 60):
                                end = 24 * 60 * 60
                            difference = end - start
                            consumption[start:end, j, k, num_patterns] = pattern[:difference]

                #bathtub
                elif appliance.__class__.__name__ == 'Bathtub':
                        #frequency is sampled from possion distribution whose average comes from survey
                        freq = np.random.poisson(households.fre_bath_day[HHID])
                        simulation_records['Bathtub']['frequency'] = freq
                        j= len(house.users)
                        for i in range(freq):
                                duration = int(pd.Timedelta(minutes=10).total_seconds())
                                simulation_records['Bathtub']['durations'].append(duration)
                                intensity = 0.2
                                prob_joint = normalize(prob_user * prob_usage)
                                u = np.random.uniform()
                                start = np.argmin(np.abs(np.cumsum(prob_joint) - u))
                                end = start + duration
                                consumption[start:end, j, k, num_patterns] = intensity
                                
            # Aggregate consumption from all users and all end-use categories
            total_consumption = xr.DataArray(data=consumption, coords=[time, users, enduse, patterns], dims=['time', 'user', 'enduse', 'patterns'])
            tot_cons = total_consumption.sum(['enduse', 'user']).mean(['patterns'])[0:-1]
            
            # Resample from 1-second to 1-hour resolution
            hourly_cons = tot_cons.resample(time='1H',label='left').sum()
            
            # Record timing information for each end-use event 
            enduse_time = total_consumption.sum(dim=['user', 'patterns'])[0:-1]
            enduse_time_records = {}
            for enduse in enduse_time['enduse'].values:
               time_indices = enduse_time.sel(enduse=enduse).where(enduse_time.sel(enduse=enduse) != 0, drop=True)['time'].values
               time_indices_formatted = [datetime.strptime(str(t)[:-3], "%Y-%m-%dT%H:%M:%S.%f").strftime("%H:%M:%S") for t in time_indices]
               enduse_time_records[enduse] = time_indices_formatted
            Tap_time = get_time_indices(enduse_time_records, 'KitchenTap')
            Wc_time = get_time_indices(enduse_time_records, 'Wc')
            Shower_time = get_time_indices(enduse_time_records, 'Shower')
            OutsideTap_time = get_time_indices(enduse_time_records, 'OutsideTap')
            WashingMachine_time = get_time_indices(enduse_time_records, 'WashingMachine')
            Dishwasher_time = get_time_indices(enduse_time_records, 'Dishwasher')
            Bathtub_time = get_time_indices(enduse_time_records, 'Bathtub')

            # Summary statistics of simulated and observed consumptions
            # Calculate sum of hourly errors between simulated and observed water use
            error = np.absolute(hourly_cons.values - observation).sum()
            #Record simulated shower duration,shower frequency and toilet frequency
            shower_d = np.sum(simulation_records['Shower']['durations'])
            shower_fre = simulation_records['Shower']['frequency']
            toilet_fre = simulation_records['Wc']['frequency']
            pickle_data = pickle.dumps(simulation_records)
            

            return {
                'Error': error,
                'Ds':shower_d/shower_fre/60,
                'Ns':shower_fre,
                'Nwc':toilet_fre,
                'Duration_freq':pickle_data,
                'Tap_time':Tap_time,
                'Wc_time':Wc_time,
                'Shower_time': Shower_time,
                'OutsideTap_time':OutsideTap_time,
                'WashingMachine_time':WashingMachine_time,
                'Dishwasher_time':Dishwasher_time,
                'Bathtub_time':Bathtub_time,
                }
#%% obtain one-day 24 hour household water consumption data
def sample_observed_data(n):
     column_means = train_sorted.iloc[n, 3:]
     observed_data = np.array(column_means)
     observe_float = observed_data.astype(float)
     return observe_float
#%%
IDlist = [0]
for HHID in IDlist:
 
  train = pd.read_excel(f"/Users/fangxinyu/Desktop/Github/HH{HHID} water meter data.xlsx")
  true_shower_d = households.dur_shower[HHID]
  true_shower_f = households.fre_shower_day[HHID]
  true_toilet_f = households.fre_flush_day[HHID]
  # iterate over training dataset
  for rownum in range(len(train)):
    df_list = []
    row = train.iloc[rownum, 2:]
    observation = sample_observed_data(rownum)

    nonzero_count = row[row != 0].count()
    #calculate hourly wateruse proportion as hourly wateruse probability
    total_sum = row.sum()
    percentages = pd.Series(row / total_sum )
    percentages = pd.concat([percentages, pd.Series([0])])
    percentages.index = pd.timedelta_range(start='00:00:00', end='24:00:00', freq='1H')
    pdf = percentages.resample('1S').ffill()[:-1]
    pdf /= np.sum(pdf)

    data = {'Error':0,'Ds':true_shower_d ,'Ns': true_shower_f,'Nwc':true_toilet_f }
    distance = AdaptivePNormDistance(
        p=1,
        # adaptive scale normalization
        scale_function=mad,
    )
    #prior distribution
    parameter_prior = Distribution(
           Wc_intensity = pyabc.RV('norm',0.042,0.015),
           tap_intensity = pyabc.RV('gamma',2.5,0,0.033),
           shower_intensity = pyabc.RV('norm',0.108,0.032),
           wm_intensity = pyabc.RV('norm',0.087,0.0315),
           dw_intensity = pyabc.RV('uniform',0.033,0.067),
           outsidetap_intensity = pyabc.RV('uniform',0.033,0.2),
          )
    abc = pyabc.ABCSMC(models = simulate_water_consumption,
                       parameter_priors=parameter_prior,
                       distance_function = distance,
                       sampler = MulticoreEvalParallelSampler(n_procs = 200),
                       population_size = 100,)

    db_path = os.path.join(tempfile.gettempdir(), f"abc_HH{HHID}_row{rownum}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
    abc.new("sqlite:///" + db_path, observed_sum_stat=data)
    history = abc.run( max_nr_populations=5,max_walltime = timedelta(hours=1.5))
