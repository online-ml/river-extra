from ast import operator
import collections
import copy
import math
import numbers
import random
import typing

import numpy as np
import time

from river import anomaly, base, compose, drift, metrics, utils


ModelWrapper = collections.namedtuple("ModelWrapper", "estimator metric")



class MICRODE(base.Estimator):
    """MESSPT 

    Using a Micro-evolutionary based on Differential Evolution for Self Parameter Tuning
    

    """

    _START_RANDOM = "random"
    _START_WARM = "warm"

    def __init__(
        self,
        estimator: base.Estimator,
        metric: metrics.base.Metric,
        params_range: typing.Dict[str, typing.Tuple],
        drift_input: typing.Callable[[float, float], float],
        grace_period: int = 500,
        drift_detector: base.DriftDetector = drift.ADWIN(),
        convergence_sphere: float = 0.001,
        seed: int = None,
        n_ind: int = 4,
        reset_one: int = 0,
        len_ind: int = 3,
        gen_to_conv: int = 5,
        F_ini: float = 0.5,
        CR_ini: float= 0.5,
        aug: float = 0.025,
        num_reset: int=1,

    ):
        super().__init__()

        #To prevent concept drift before convergence, always a random option is included in each run. 
        #If random option is the best after gp_init_rand grace periods, reset
        #If reset_one is 0, not to include this random option in the algorithm. Else yes

        self.random_option = None
        self._old_b_rand_opt = None
        self.reset_one = reset_one
        self._gp_init_rand = num_reset
        self._random_best = False
        #If rand is best is 0, currently the random option is not the best. 1 else
        self._rand_is_best = 0

        ##To measure cost
        ##
        self._time_to_conv = []
        self._time_acc = 0
        self._num_ex = 0
        self._num_ex_array = []
        self._num_models = 0
        self._num_models_array = []
        ##

        self.len_ind = len_ind
        self.estimator = estimator
        self.metric = metric
        self.params_range = params_range
        self.drift_input = drift_input
        self._converged = False
        self.grace_period = grace_period
        self.drift_detector = drift_detector
        self.convergence_sphere = convergence_sphere
        #self.first_pop = True
        self.seed = seed
        self._rng = random.Random(self.seed)
        self._n = 0
        #To check differences between current and old best
        self._old_b = None
        #number of current generations with same (or low differences) individual as best
        self._gen_same_b = 0
        #number of generations (max) to consider convergence
        self._gen_to_conv = gen_to_conv
        #number of individuals into pop
        self.n_ind = n_ind
        #probability of mutation
        self.current_pop = []

        self._best_i = None
        self._best_child = False
        
        self.current_children = []
        self._best_estimator = None
        
        #F and CR values control mutation and cross. If is_adaptive is True, they will change dynamically.
        #Else, they will be static values during all the process

        self.is_adaptive = True
        #How F and CR changes after each grace period
        self.aug = aug

        #initial values of F and CR
        self.F_ini = F_ini
        self.CR_ini = CR_ini
        self.F = self.F_ini 
        self.CR = self.CR_ini 

        #Create first population
        self._create_pop(self.n_ind)

    #Generate new gen. It could be int, float or discrete. Discrete option is not tested
    def __generate(self, p_data):
        p_type, p_range = p_data
        if p_type == int:
            return self._rng.randint(p_range[0], p_range[1])
        elif p_type == float:
            return self._rng.uniform(p_range[0], p_range[1])
        #DISCRET (Not tested)
        ''' 
        elif p_type == "discrete":
            choice = self._rng.randint(0, len(p_range))
            return p_range[choice]
        '''

    #Generate a new random configuration

    def _random_config(self):
        return self._recurse_params(
            operation = "generate",
            p_data=self.params_range,
            e1_data=self.estimator._get_params()
        )

    #Create a new wrapper (individual + metric)

    def _create_wrapper(self, ind, metric, est):
        w = ModelWrapper(est.clone(ind, include_attributes=True), metric)
        return w


    #Create a new pop o n individuals from scratch

    def _create_pop(self, n):
        self.current_pop = [None]*n
        for i in range(n):
            self.current_pop[i] = self._create_wrapper(self._random_config(), 
                self.metric.clone(include_attributes=True), self.estimator)

    #Scale for checking similarities between best individuals (old and current)

    def _normalize_flattened_hyperspace(self, orig):
        scaled = {}
        self._recurse_params(
            operation="scale",
            p_data=self.params_range,
            e1_data=orig,
            prefix="",
            scaled=scaled
        )
        return scaled

    @property
    def best(self):
        
        return self._best_estimator

    @property
    def random_best(self):
        return self._random_best
    


    @property

    #If best and old best are too similar, we consider they are equal in this generations.
    #If they are equal (low differences) after gen_to_conv generations, we consider convergence

    def _models_converged(self) -> bool:
        # Normalize params to ensure they contribute equally to the stopping criterion
        
        scaled_params_b = self._normalize_flattened_hyperspace(
                self._best_estimator._get_params()
            )
        r_sphere=1
        
        if self._old_b != None:
            r_sphere = utils.math.minkowski_distance(scaled_params_b, 
                self._old_b, p=2)

        #If low differences:

        if r_sphere < self.convergence_sphere:
            #Number of generations in which they are similar is the maximum (convergence)
            if self._gen_same_b == self._gen_to_conv:
                self._gen_same_b = 0
                self._old_b = None
                return True
            else:
                #one more generation in which current and old are similar
                self._gen_same_b = self._gen_same_b+1
                return False
        #Consider differences (0 generations without similarity)
        else:
            self._gen_same_b = 0
            self._old_b = scaled_params_b
            return False


    #Combine parameters from 3 individuals

    def __combine(self, p_info, param1, param2, param3, func):

        p_type, p_range = p_info
        new_val = func(param1, param2, param3)

        # Range sanity checks
        if new_val < p_range[0]:
            new_val = p_range[0]
        if new_val > p_range[1]:
            new_val = p_range[1]

        new_val = round(new_val, 0) if p_type == int else new_val
        return new_val

    
    #Combine 3 estimators to generate one new. new = best + F(r1-r2)
    def _gen_new_estimator(self, e1, e2, e3, func):
        """Generate new configuration given two estimators and a combination function."""
        e1_p = e1.estimator._get_params()
        e2_p = e2.estimator._get_params()
        e3_p = e3.estimator._get_params()


        new_config = self._recurse_params(
            operation="combine",
            p_data=self.params_range,
            e1_data=e1_p,
            func=func,
            e2_data=e2_p,
            e3_data=e3_p,
        )
        # Modify the current best contender with the new hyperparameter values
        new = ModelWrapper(
            copy.deepcopy(self._best_estimator),
            self.metric.clone(include_attributes=True),

        )
        
        new.estimator.mutate(new_config)

        return new

    def __combine_cross_rate(self, p_info, param1, param2, num, index_change, gen, func):

        p_type, p_range = p_info
        new_val = func(param1, param2, num, index_change, gen)

        # Range sanity checks
        if new_val < p_range[0]:
            new_val = p_range[0]
        if new_val > p_range[1]:
            new_val = p_range[1]

        new_val = round(new_val, 0) if p_type == int else new_val
        return new_val

    def _gen_new_estimator_cross_rate(self, e1, e2, index_change, func):
        """Generate new configuration given two estimators and a combination function."""

        e1_p = e1.estimator._get_params()
        e2_p = e2.estimator._get_params()


        new_config = self._recurse_params(
            operation="combine_cr",
            p_data=self.params_range,
            e1_data=e1_p,
            index_change=index_change,
            func=func,
            e2_data=e2_p,
            gen=0
        )
        # Modify the current best contender with the new hyperparameter values
        new = ModelWrapper(
            copy.deepcopy(e2.estimator),
            self.metric.clone(include_attributes=True),

        )
        new.estimator.mutate(new_config)

        return new


    def __flatten(self, prefix, scaled, p_info, e_info):
        _, p_range = p_info
        interval = p_range[1] - p_range[0]
        scaled[prefix] = (e_info - p_range[0]) / interval


    #Generate: Generate a new random individual
    #Scale: Scale for checking similarities
    #Combine: Combine 3 individuals
    #Combine_cr: Cross two individuals: the current one and the obtained from combinated (mutation) 3 individuals.
    #To combine_cr: gen by gen check if include the gen from the current individual or from the mutation.  


    def _recurse_params(
        self, operation, p_data, e1_data, *, index_change=None, 
        func=None, e2_data=None, e3_data=None,
         prefix=None, scaled=None, gen=None
    ):  

        # Sub-component needs to be instantiated
        if isinstance(e1_data, tuple):
            sub_class, sub_data1 = e1_data

            if operation=="combine_cr":
                _, sub_data2 = e2_data
                sub_data3 = {}

            elif operation == "combine":
                _, sub_data2 = e2_data
                _, sub_data3 = e3_data

            else:
                sub_data2 = {}
                sub_data3 = {}


            sub_config = {}
            for sub_param, sub_info in p_data.items():
                if operation == "scale":
                    sub_prefix = prefix + "__" + sub_param
                else:
                    sub_prefix = None
                sub_config[sub_param] = self._recurse_params(
                    operation=operation,
                    p_data=sub_info,
                    e1_data=sub_data1[sub_param],
                    func=func,
                    e2_data=sub_data2.get(sub_param, None),
                    e3_data=sub_data3.get(sub_param, None),
                    index_change=index_change,
                    prefix=sub_prefix,
                    scaled=scaled,
                    gen=gen
                )
            return sub_class(**sub_config)

        # We reached the numeric parameters
        if isinstance(p_data, tuple):
            if operation == "generate":
                return self.__generate(p_data)
            if operation == "scale":
                self.__flatten(prefix, scaled, p_data, e1_data)
                return
            if operation == "combine_cr":
                num = self._rng.uniform(0,1)
                res = self.__combine_cross_rate(p_data, e1_data, e2_data, num, 
                    index_change, gen, func)
                return res

            # combine
            #NEW
            p_type, p_range = p_data
            if p_type == int or p_type == float:
                return self.__combine(p_data, e1_data, e2_data, e3_data, func)
            else:
                return self.__combine(p_data, e1_data, e2_data, e3_data, func)# TODO: func_disc
        # The sub-parameters need to be expanded
        config = {}
        for p_name, p_info in p_data.items():
            e1_info = e1_data[p_name]

            if operation == "combine":
                e2_info = e2_data[p_name]
                e3_info = e3_data[p_name]

            elif operation == "combine_cr":
                e2_info = e2_data[p_name]
                e3_info = {}

            else:
                e2_info = {}
                e3_info = {}

            if operation == "scale":
                sub_prefix = prefix + "__" + p_name if len(prefix) > 0 else p_name
            else:
                sub_prefix = None

            if not isinstance(p_info, dict):
                config[p_name] = self._recurse_params(
                    operation=operation,
                    p_data=p_info,
                    e1_data=e1_info,
                    func=func,
                    e2_data=e2_info,
                    e3_data=e3_info,
                    index_change=index_change,
                    prefix=sub_prefix,
                    scaled=scaled,
                    gen=gen
                )
            else:
                sub_config = {}
                for sub_name, sub_info in p_info.items():
                    if operation == "scale":
                        sub_prefix2 = sub_prefix + "__" + sub_name
                    else:
                        sub_prefix2 = None
                    sub_config[sub_name] = self._recurse_params(
                        operation=operation,
                        p_data=sub_info,
                        e1_data=e1_info[sub_name],
                        func=func,
                        e2_data=e2_info.get(sub_name, None),
                        e3_data=e3_info.get(sub_name, None),
                        index_change=index_change,
                        prefix=sub_prefix2,
                        scaled=scaled,
                        gen=gen
                    )
                    if operation == "combine_cr":
                        gen = gen+1
                config[p_name] = sub_config
        return config


    #Combine for discrete. It is not tested!
    
    '''
    def func_disc(self, v1, v2, v3):
        r_f_c = self._rng.uniform(0,1)
        to_c = v2 if r_f_c <= 0.5 else v3
        r_f_c_2 = self._rng.uniform(0,1)
        res = to_c if r_f_c_2 < self.F/(1+self.F) else v1
        return res
    '''

    #Cross operator from differential evolution. Best_1 operator: new = best + F(r1-r2)


    def _de_cross_best_1(self, i):
        #Select 2 individuals from current list. It has to be different from the current (i) individual
        list_el = np.ndarray.tolist(np.arange(1,self.n_ind))
        if i!=0:
            list_el.remove(i)
        r1, r2 = self._rng.sample(list_el, 2)
        #best in 0 pos
        n_p = self._gen_new_estimator(
            self.current_pop[0], self.current_pop[r1], self.current_pop[r2],
             lambda h1, h2, h3: h1 + self.F*(h2-h3)
        )
        return n_p

    def _ev_op_crossover(
        self, operation, ind=None, index=None):
        
        return self._de_cross_best_1(index)


    #To end cross. Check if new gen will be obtained from current individual or mutation

    def func_cr(self, p1, p2, num, index_change, gen):
                
        if num<self.CR or index_change==gen:
            return p1
        else:
            return p2

    #Cross current populations and generate a children population

    def _cross_current_pop(self):
        for i in range(self.n_ind):
            current_el = self.current_pop[i]
            new_c = self._ev_op_crossover("de_best_1", None, i)
            index_change = self._rng.randint(0, self.len_ind-1)
            
            new_c = self._gen_new_estimator_cross_rate(new_c, current_el,
                index_change, self.func_cr)
            self.current_children.append(new_c)
    

    def _learn_one(self, wrap, x, y):
        wrap.estimator.learn_one(x,y)

    def _update(self, wrap, x, y):
        scorer = getattr(wrap.estimator, "predict_one")


        y_pred = scorer(x)
        
        wrap.metric.update(y, y_pred)

    def _sort_c_pop(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self.current_pop.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self.current_pop.sort(key=lambda mw: mw.metric.get())

    def _sort_c_children(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self.current_children.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self.current_children.sort(key=lambda mw: mw.metric.get())

    

    def _generate_next_current_pop(self):
        
        #Update all individuals to random except for the best one (i=0). 
        #To do this, clone the best model for all individuals, and change configuration of all except
        #for the best
        for i in range(self.n_ind):
            
            self.current_pop[i] = ModelWrapper(copy.deepcopy(self._best_i.estimator),
            self.metric.clone(include_attributes=True)

            )
            if i!=0:
                self.current_pop[i].estimator.mutate(self._random_config())
            
        #Check if currently rand is best. If it is the best for k periods, we consider a restart

        if self.reset_one==1:
            

            #If current one is not the best, next random option will be (of course), random
            if self._rand_is_best==0:
                self.random_option = self._create_wrapper(self._random_config(), 
                        self.metric.clone(include_attributes=True), self.estimator)

            else:
                #If currently random is the best option, keep this option (to check if it continues 
                #being the best for next grace periods and if we consider restarting or not)
                self.random_option = ModelWrapper(copy.deepcopy(self.random_option.estimator),
            self.metric.clone(include_attributes=True))


    #Learn a new instance if converged:
    def _learn_converged(self, x, y):
        scorer = getattr(self._best_estimator, "predict_one")
        y_pred = scorer(x)

        input = self.drift_input(y, y_pred)
        self.drift_detector.update(input)

        # Drift detected -> We need to start the optimization process from scratch
        if self.drift_detector.drift_detected:
            self.F = self.F_ini
            self.CR = self.CR_ini
            # if self.reset_one==1:
            #     print("drift detected mder")
            # else:
            #     print("drift detected mde")

            self._n = 0
            self._converged = False
            self._best_child = False
            
            #Except for the current best (i=0), all the others are restarted from scratch

            for i in range(1, self.n_ind):
                self.current_pop[i] = self._create_wrapper(self._random_config(), 
                    #self._best_i.metric.clone(include_attributes=True)
                    self.metric.clone(include_attributes=True), self.estimator

                )
            
            #Current best is kept in the population restarted
            self.current_pop[0] = ModelWrapper(copy.deepcopy(self._best_estimator),
                self.metric.clone(include_attributes=True))
            



            # There is no proven best model right now
            self._best_i = None
            self._best_estimator = None
    
            #Variables related to random option
            self.random_option = None
            self._random_best = None

            #Variables related to cost
            self._time_acc = 0
            self._num_ex = 0
            self._num_models = 0

            return


        #If concept drift not detected
        self._best_estimator.learn_one(x, y)

   
    #Learn one. If converged, call to _learn converged. Else:
    '''
    Else: Update and learn with each individual for this data. If we reach a grace period (number)
    of evaluated = grace_period -> Run micro evolutionary algorithm

    '''
    def learn_one(self, x, y):
        
        if self._converged ==True:
            self._n = self._n + 1
            self._learn_converged(x, y)
        else:
            #num ex not converged  
            t1 = time.time()
            self._num_ex = self._num_ex + 1
            
            #update and learn individuals of pop
            for wrap in self.current_pop:
                self._update(wrap, x, y)

                self._learn_one(wrap, x, y)

            self._sort_c_pop()

            #if current children exists: update and learn for each individual
            if self.current_children != []:
                for wrap in self.current_children:
                    self._update(wrap, x, y)
                    self._learn_one(wrap, x, y)
                self._sort_c_children()

            #select the best between current pop, current children pop and the random option
            #######
            ####
            if self.current_children != []:
                if self.metric.bigger_is_better:
                    if self.current_pop[0].metric.get() < self.current_children[0].metric.get():
                        self._best_i = self.current_children[0]
                        self._best_child = True
                    else:
                        self._best_i = self.current_pop[0]
                        self._best_child = False
                else:
                    if self.current_pop[0].metric.get() > self.current_children[0].metric.get():
                        self._best_i = self.current_children[0]
                        self._best_child = True
                    else:
                        self._best_i = self.current_pop[0]
                        self._best_child = False
            else:
                self._best_i = self.current_pop[0]
                self._best_child = False


            self._random_best = False
            if self.random_option != None:
                #update and learn for the random option
                self._update(self.random_option, x, y)
                self._learn_one(self.random_option, x, y)
                #check if random is the best
                if self.metric.bigger_is_better:

                    if self.random_option.metric.get() > self._best_i.metric.get():
                        

                        self._random_best = True
                else:
                    if self.random_option.metric.get() < self._best_i.metric.get():
                        

                        
                        self._random_best = True
            ####
            #######

            
            self._best_estimator = self._best_i.estimator

            self._n = self._n + 1
            #if reach grace period. Run micro-ev algorithm
            if self._n % self.grace_period == 0:
                #If random option is the best
                if self._random_best == True:
                    #restart values for micro-ev algorithm to look for better options.
                    self._num_models = self._num_models + self.n_ind*2+1
                    self.F = self.F_ini
                    self.CR = self.CR_ini
                    self._rand_is_best = self._rand_is_best+1
                    #If the number of grace periods (in a row) in which random is the best is > than
                    # gp_init_rand --> Restart 
                    if self._rand_is_best >= self._gp_init_rand:
                         
                        for i in range(2, self.n_ind):
                            self.current_pop[i] = self._create_wrapper(self._random_config(), 
                                self.metric.clone(include_attributes=True), self.estimator

                            )
                        
                        self.current_pop[0] = ModelWrapper(copy.deepcopy(self.random_option.estimator),
                            self.metric.clone(include_attributes=True))
                    
                        #copy best before rand
                        self.current_pop[1] = ModelWrapper(copy.deepcopy(self._best_i.estimator),
                            self.metric.clone(include_attributes=True))

                        self._old_b_rand_opt = None
                        self.current_children = []
                        self.F = self.F_ini
                        self.CR = self.CR_ini
                        self._n = 0
                        self._converged = False
                        self._best_child = False
                        self._random_best = False
                        self.random_option = None

                        # There is no proven best model right now
                        self._best_i = None
                        self._best_estimator = None
                        self._rand_is_best = 0

                    #If not reached the gp_init_rand number: run the micro-ev algorithm

                    else:

                        if self.is_adaptive:
                            #to fast conv
                            self.F = self.F - self.aug
                            self.CR = self.CR + self.aug
                            if self.F < 0.0:
                                self.F = 0.0
                            if self.CR > 1:
                                self.CR = 1
                        
                        self._generate_next_current_pop()
                        self.current_children=[]
                        self._best_child = False
                        
                        #Check convergence

                        if self._models_converged:
                            t2 = time.time()
                            # if self.reset_one==0:
                            #     print("mde convergence")
                            # else:
                            #     print("mder convergence")
                            self._time_acc = self._time_acc + (t2-t1)
                            self._time_to_conv.append(self._time_acc)
                            self._num_ex_array.append(self._num_ex)
                            self._num_models_array.append(self._num_models)
                            self._converged = True
                        else:
                            self._cross_current_pop()



                #We are in the first grace period. Just cross the current pop, not create a new one
                elif self._n / self.grace_period == 1:
                    self._num_models = self._num_models + self.n_ind
                    self._rand_is_best = 0
                    
                    #firstly, models are mixed
                    self._cross_current_pop()
                    
                    for j in range(self.n_ind):
                        self.current_pop[j] = ModelWrapper(
                        copy.deepcopy(self.current_pop[j].estimator),
                            self.metric.clone(include_attributes=True),

                            )

                #Most ordinary case: Random is not the best and not the first generation: Run the micro_ev
                #algorithm

                else:
                    self._rand_is_best = 0
                    self._num_models = self._num_models + self.n_ind*2+1

                    if self.is_adaptive:
                        #to fast conv
                        self.F = self.F - self.aug
                        self.CR = self.CR + self.aug
                        if self.F < 0.0:
                            self.F = 0.0
                        if self.CR > 1:
                            self.CR = 1
                    
                    self._generate_next_current_pop()
                    self.current_children=[]
                    self._best_child = False
                    
                    if self._models_converged:
                        t2 = time.time()
                        # if self.reset_one==0:
                        #     print("mde convergence")
                        # else:
                        #     print("mder convergence")
                        self._time_acc = self._time_acc + (t2-t1)
                        self._time_to_conv.append(self._time_acc)
                        self._num_ex_array.append(self._num_ex)
                        self._num_models_array.append(self._num_models)
                        self._converged = True
                    else:
                        self._cross_current_pop()
                                        
                

            t2 = time.time()
            self._time_acc = self._time_acc + (t2-t1)
        return self 


    def predict_one(self, x, **kwargs):

        if self._random_best:
            return self.random_option.estimator.predict_one(x, **kwargs)


        elif self._best_estimator == None:
            self._sort_c_pop()
            return self.current_pop[0].estimator.predict_one(x, **kwargs)
        
        else:
            return self._best_estimator.predict_one(x, **kwargs)


    @property
    def converged(self):
        return self._converged
    

    #To measure cost
        
    @property
    def time_to_conv(self):
        if self.converged:
            return self._time_to_conv
        else:
            self._time_to_conv.append(self._time_acc)
            return self._time_to_conv
    
    @property
    def num_ex(self):
        if self.converged:
            return self._num_ex_array
        else:
            self._num_ex_array.append(self._num_ex)
            return self._num_ex_array

    @property
    def num_models(self):
        if self.converged:
            return self._num_models_array
        else:
            self._num_models_array.append(self._num_models)
            return self._num_models_array