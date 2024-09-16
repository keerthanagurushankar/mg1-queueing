import heapq
import random, math
import numpy as np
import lib

class Event:
    def __init__(self, event_type, time, job=None):
        self.event_type = event_type
        self.time = time
        self.job = job

    def __lt__(self, other):
        return self.time < other.time

class Job:
    def __init__(self, arrival_time, service_time, job_class):
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.remaining_time = service_time        
        self.job_class = job_class
        self.priority = job_class.priority(self.remaining_time, self.service_time, 0)

    def __lt__(self, other):
        if self.priority == other.priority: # isclose?
            return self.arrival_time < other.arrival_time # older is better within class
        return self.priority > other.priority # higher priority is better

class JobClass:
    def __init__(self, index, l, S): 
        self.index = index
        self.l = l
        self.generate_service_time = S
        self.priority = None

    def generate_next_job(self, current_time):
        return Job(current_time + random.expovariate(self.l),
                    self.generate_service_time(), self)

class MG1:
    def __init__(self, job_classes, policy):
        # Parameters
        self.job_classes = job_classes
        self.policy = policy    
        self.simulation_time = 10**6
        self.inspection_rate = 0.1
        
        # initialize priority_function of job classes
        for job_class in job_classes:
            if policy.priority_fn == None:
                job_class.priority = lambda r, s, t, idx=job_class.index: -idx
            else:
                job_class.priority = lambda r, s, t, idx=job_class.index: \
                   policy.priority_fn(r, s, t, idx)

        # System state
        self.event_queue = [] # holds events in order of event time
        self.job_queue = [] # holds waiting jobs in order of priority
        self.current_time = 0
        self.current_job = None # if busy, is job being served
        self.current_service_start_time = None # if busy, is time of start of last service
        self.current_departure_event = None # departure event of job being served

        # Metrics
        self.metrics = []
        
    def initialize(self):     
        for job_class in self.job_classes:
            job = job_class.generate_next_job(0)
            heapq.heappush(self.event_queue, Event('Arrival', job.arrival_time, job))
        
    def run(self):
        self.initialize()
        
        while self.event_queue and self.current_time < self.simulation_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == 'Arrival':           
                self.handle_arrival(event)
            elif event.event_type == 'Departure':                  
                self.handle_departure(event)
            elif event.event_type == 'Inspection':
                self.handle_inspection(event)

    def handle_arrival(self, event):
        job = event.job
        heapq.heappush(self.job_queue, job)     
        
        # Schedule next arrival for the appropriate job_class
        next_job = job.job_class.generate_next_job(self.current_time)
        heapq.heappush(self.event_queue, Event('Arrival', next_job.arrival_time, next_job))

        # Start working on the arrival if idle or lower preemptive priority
        if self.current_job is None:
            self.start_service()
        elif self.policy.is_preemptive: # and self.current_job.priority < job.priority:
            self.update_priorities()
            if self.current_job.priority < job.priority:
                self.preempt_current_job()

    def handle_departure(self, event):
        assert event.job == self.current_job, "Tried to depart job not in service"
        self.record_metrics(event.job, self.current_time)

        if self.job_queue:
            self.start_service()
        else:
            self.current_job = None
            self.current_service_time = None
            self.current_departure_event = None

    def start_service(self):
        self.update_priorities()
        self.current_job = heapq.heappop(self.job_queue)
        self.current_service_start_time = self.current_time
        
        departure_time = self.current_time + self.current_job.remaining_time
        assert self.current_time > self.current_job.arrival_time or math.isclose( \
          self.current_time, self.current_job.arrival_time),"Tried to start service before arrival"
        
        self.current_departure_event = Event('Departure', departure_time, self.current_job)
        heapq.heappush(self.event_queue, self.current_departure_event)

    def update_priorities(self):
        if not self.policy.is_dynamic_priority:
            return
        
        updated_priority = False
        for job in self.job_queue:
            old_priority = job.priority
            job.priority = job.job_class.priority(job.remaining_time, job.service_time,
                                                  self.current_time - job.arrival_time)
            if not math.isclose(old_priority, job.priority):
                updated_priority = True
        if updated_priority:
            heapq.heapify(self.job_queue)
        
    def preempt_current_job(self):
        time_in_service = self.current_time - self.current_service_start_time
        self.current_job.remaining_time = self.current_job.remaining_time - time_in_service
        heapq.heappush(self.job_queue, self.current_job)
        
        self.event_queue.remove(self.current_departure_event)
        heapq.heapify(self.event_queue)
        
        self.start_service()
 
    def record_metrics(self, job, departure_time):
        job_metrics = {'job_class': job.job_class.index, 
                       'arrival_time': job.arrival_time,
                       'departure_time': departure_time,
                       'job_size': job.service_time,
                       'response_time': departure_time - job.arrival_time,
                       'waiting_time': departure_time - job.arrival_time - job.service_time,
                       'priority': job.priority} # at completion
        #print(job_metrics)
        self.metrics.append(job_metrics)

    def handle_inspection(self):        
        # check that highest priority job is being worked on
        current_job_priority = self.current_job.job_class.priority(
            self.current_time - self.current_service_start_time,
            self.current_job.service_time,
            self.current_time - job.arrival_time)
        
        for job in self.job_queue: 
            job_priority = job.job_class.priority(job.remaining_time, job.service_time,
                                            self.current_time - job.arrival_time)

            assert job_priority < current_job_priority or \
                (not self.is_preemptive and job.arrival_time > self.current_service_start_time)

