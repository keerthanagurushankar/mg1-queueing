import heapq
from collections import deque
import math
import json, os

import logging
logging.basicConfig(level=logging.INFO)

from simulations import Job, JobClass

class MG1:
    def __init__(self, job_classes, policy, simulation_time=10**5, time_step=0.01, ClassFCFS=True):
        # Parameters
        self.job_classes = job_classes
        self.policy = policy
        
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.ClassFCFS = ClassFCFS
        
        # initialize priority_function of job classes
        for k, job_class in enumerate(job_classes):
            if policy.priority_fn == None:
                job_class.priority = lambda r, s, t, idx=job_class.index: -idx
            elif type(policy.priority_fn) is list:
                job_class.priority = policy.priority_fn[k]
            else:
                job_class.priority = lambda r, s, t, idx=job_class.index: policy.priority_fn(r, s, t, idx)

        # System state
        self.nextArrivals = [] #holds the next job to arrive of each class
        if ClassFCFS:
            self.classQueues = {job_class: deque() for job_class in job_classes}
        self.job_queue = [] # holds waiting jobs in order of priority
        self.current_time = 0
        self.current_job = None

        # Metrics
        self.metrics = []
        
    def initialize(self):     
        for job_class in self.job_classes:
            job = job_class.generate_next_job(0)
            self.nextArrivals.append(job)
        
    def run(self):
        self.initialize()
        while self.current_time < self.simulation_time:
            self.current_time += self.time_step
            self.handle_arrivals()
            self.get_current_job()
            self.step_service()

    def handle_arrivals(self):
    # Any jobs that have arrived by current_time pushed in queue and replaced by next arrival
        arrivingJobs = filter(lambda j : j.arrival_time <= self.current_time, self.nextArrivals)
        for job in arrivingJobs:
            if self.ClassFCFS:
                if not self.classQueues[job.job_class]:
                    heapq.heappush(self.job_queue, job)
                self.classQueues[job.job_class].append(job)
            else: heapq.heappush(self.job_queue, job)

            self.nextArrivals.remove(job)
            nextJob = job.job_class.generate_next_job(self.current_time)
            self.nextArrivals.append(nextJob)
    
    def update_priorities(self):
    # Copied from simulations.py, updates all priorities
        if not self.policy.is_dynamic_priority:
            return
        
        updated_priority = False
        for job in self.job_queue:
            new_priority = job.current_priority(self.current_time)
            if not math.isclose(new_priority, job.priority):
                logging.debug(f"I'm updating priority of {job.job_class.index, job.arrival_time} from {job.priority} to {new_priority}")                
                updated_priority = True
                job.priority = new_priority                

        if self.current_job:
            self.current_job.priority = self.current_job.current_priority(self.current_time)
       
        if updated_priority:
            heapq.heapify(self.job_queue)
    
    def get_current_job(self):
    # Updates current_job
        if self.job_queue:
            if self.current_job is None:
                self.update_priorities()
                self.current_job = heapq.heappop(self.job_queue)
            elif self.policy.is_preemptive:
                self.update_priorities()
                heapq.heappush(self.job_queue, self.current_job)
                self.current_job = heapq.heappop(self.job_queue)
    
    def step_service(self):
    # Serves current_job for time_step, potentially departing it
        if self.current_job is None: return
        assert self.current_time >= self.current_job.arrival_time, "Tried to start service before arrival"
        self.current_job.remaining_time -= self.time_step
        if self.current_job.remaining_time <= 0.0:
            self.record_metrics(self.current_job, self.current_time)
            if self.ClassFCFS:
                assert self.current_job == self.classQueues[self.current_job.job_class].popleft()
                if self.classQueues[self.current_job.job_class]:
                    nextJob = self.classQueues[self.current_job.job_class][0]
                    heapq.heappush(self.job_queue, nextJob)
            self.current_job = None
    
    def record_metrics(self, job, departure_time):
    # Copied from simulations.py, records metrics of a job on departure
        job_metrics = {'job_class': job.job_class.index, 
                       'arrival_time': job.arrival_time,
                       'job_size': job.service_time,
                       'departure_time': departure_time,                       
                       'response_time': departure_time - job.arrival_time,
                       'waiting_time': departure_time - job.arrival_time
                           - job.service_time,
                       'priority': job.priority} # at completion
        self.metrics.append(job_metrics)

    def save_metrics(self, path):
    # Copied from simulations.py, saves metrics in json file
        os.makedirs(path, exist_ok=True)
        json.dump(self.metrics, open(os.path.join(path, "metrics.json"), 'w'))
            
        for job_class in self.job_classes:
            arrival_sequence = [job for job in self.metrics
                                if job['job_class'] == job_class.index]
            fname = os.path.join(path, f"arrival_sequence{job_class.index}.json")
            json.dump(arrival_sequence, open(fname, 'w'))
        
        logging.info(f"Saved metrics to {path}")