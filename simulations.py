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

    def current_priority(self, current_time):
        return self.job_class.priority(self.remaining_time, self.service_time,
                                                current_time - self.arrival_time)

class JobClass:
    def __init__(self, index, l, S): 
        self.index = index
        self.l = l
        self.generate_service_time = S
        self.priority = None
        self.b = None

    def generate_next_job(self, current_time):
        return Job(current_time + random.expovariate(self.l),
                    self.generate_service_time(), self)

class MG1:
    def __init__(self, job_classes, policy):
        # Parameters
        self.job_classes = job_classes
        self.policy = policy    
        self.simulation_time = 10**5
        self.inspection_rate = 1
        
        # initialize priority_function of job classes
        for job_class in job_classes:
            if policy.priority_fn == None:
                job_class.priority = lambda r, s, t, idx=job_class.index: -idx
            else:
                job_class.priority = lambda r, s, t, idx=job_class.index: \
                   policy.priority_fn(r, s, t, idx)
                job_class.b = policy.priority_fn(0, 0, 1, job_class.index) - policy.priority_fn(0, 0, 0, job_class.index)

        # System state
        self.event_queue = [] # holds events in order of event time
        self.job_queue = [] # holds waiting jobs in order of priority
        self.current_time = 0
        self.current_job = None # if busy, is job being served
        self.current_service_start_time = None # if busy, is time of start of last service
        self.current_departure_event = None # departure event of job being served
        self.current_preemption_event = None

        # Metrics
        self.metrics = []
        
    def initialize(self):     
        for job_class in self.job_classes:
            job = job_class.generate_next_job(0)
            heapq.heappush(self.event_queue, Event('Arrival', job.arrival_time, job))

        heapq.heappush(self.event_queue, Event('Inspection', random.expovariate(self.inspection_rate)))
        
    def run(self):
        self.initialize()
        
        while self.event_queue and self.current_time < self.simulation_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            print(event.time, event.event_type, event.job.job_class.b if event.job else None)
            
            if event.event_type == 'Arrival':           
                self.handle_arrival(event)
            elif event.event_type == 'Departure':                  
                self.handle_departure(event)
            elif event.event_type == 'Inspection':
                self.handle_inspection()
            elif event.event_type == 'PreemptionCheck':
                self.preempt_current_job()

    def handle_arrival(self, event):
        job = event.job
        heapq.heappush(self.job_queue, job)     
        
        # Schedule next arrival for the appropriate job_class
        next_job = job.job_class.generate_next_job(self.current_time)
        heapq.heappush(self.event_queue, Event('Arrival', next_job.arrival_time, next_job))

        # Start working on the arrival if idle or lower preemptive priority
        if self.current_job is None:
            self.start_service()
        elif self.policy.is_preemptive: 
            self.update_priorities()
            if self.current_job.priority < job.priority:
                self.preempt_current_job() # the arrival has higher priority so we should preempt
            else:
                self.schedule_preemption_check(job) # the arrival may acquire higher priority later

    def handle_departure(self, event):
        assert event.job == self.current_job, "Tried to depart job not in service"
        self.record_metrics(event.job, self.current_time)

        self.current_job = None
        self.current_service_start_time = None
        self.current_departure_event = None
        
        if self.job_queue:
            self.update_priorities()            
            self.start_service()

    def handle_inspection(self):
        if self.current_job is None: 
            assert not self.job_queue, "if idle, must not have waiting jobs"
        else:
            # check that highest priority job is being worked on
            if self.policy.is_preemptive:
                current_job_priority = self.current_job.current_priority(self.current_time)

                for job in self.job_queue:
                    job_priority = job.current_priority(self.current_time)
                    assert job_priority <= current_job_priority, "Not working on highest prio job" + f"{job_priority, current_job_priority, self.current_service_start_time, job.arrival_time}"
            else:
                
                for job in self.job_queue:
                    assert job.priority <= self.current_job.priority or job.arrival_time \
                       >= self.current_service_start_time, "Not working on highest prio job"

        next_inspection_time = self.current_time + random.expovariate(self.inspection_rate)
        heapq.heappush(self.event_queue, Event('Inspection', next_inspection_time))

    def start_service(self):
        # assume system is not empty and job priorities are up to date
        # start working on job of current highest priority
        self.current_job = heapq.heappop(self.job_queue)
        self.current_service_start_time = self.current_time
        
        departure_time = self.current_time + self.current_job.remaining_time
        assert self.current_time >= self.current_job.arrival_time, "Tried to start service before arrival"
        
        self.current_departure_event = Event('Departure', departure_time, self.current_job)
        heapq.heappush(self.event_queue, self.current_departure_event)

        print("Starting job of ", self.current_job.priority, self.current_job.job_class.b, " at ", self.current_time)
        for job in self.job_queue:
            print(job.job_class.b, job.priority, job.arrival_time)

        self.schedule_preemption_check()

    def update_priorities(self):
        if not self.policy.is_dynamic_priority:
            return

        updated_priority = False
        for job in self.job_queue:
            new_priority = job.current_priority(self.current_time)
            if not math.isclose(new_priority, job.priority):
                updated_priority = True
                job.priority = new_priority

        if self.current_job:
            self.current_job.priority = self.current_job.current_priority(self.current_time)
       
        if updated_priority:
            heapq.heapify(self.job_queue)
            print("I updated priorities at ", self.current_time)

    def preempt_current_job(self):
        # if busy, put currently served job in queue, update and start work on whatever is the new highest priority job
        if self.current_job is None:
            return
        
        time_in_service = self.current_time - self.current_service_start_time
        self.current_job.remaining_time = self.current_job.remaining_time - time_in_service
        heapq.heappush(self.job_queue, self.current_job)
        
        self.event_queue.remove(self.current_departure_event)
        heapq.heapify(self.event_queue)

        self.update_priorities()
        self.start_service()

    def calculate_overtake_time(self, queue_job):
        b_curr, b_queue = self.current_job.job_class.b, queue_job.job_class.b
        if b_queue <= b_curr:
            return None

        t_overtake = (b_queue*queue_job.arrival_time - b_curr*self.current_job.arrival_time) / (b_queue-b_curr)
        return t_overtake

    def schedule_preemption_check(self, new_job = None):
        # check if new arrival or some job in queue may preempt the current service
        preemption_event = None
        
        if new_job:
            # if a higher b job just arrived, it may preempt the low b job if it hasn't completed service
            t_overtake = self.calculate_overtake_time(new_job)
            if t_overtake and t_overtake > self.current_time:
                preemption_event = Event('PreemptionCheck', t_overtake, new_job)
        else:
            # when starting new service, check and schedule if any job in queue may grow to overtake priority
            print("I'm checking to schedule preemptions. The time is now ", self.current_time, self.current_job.job_class.b)
            overtake_times = []
            for job in self.job_queue:
                t_overtake = self.calculate_overtake_time(job)
                print("job in queue", job.job_class.b, t_overtake, job.arrival_time)
                if t_overtake and t_overtake > self.current_time:
                    overtake_times.append(t_overtake)
                    
            if overtake_times:
                next_preemption_time = min(overtake_times)
                preemption_event = Event('PreemptionCheck', next_preemption_time)
                print("I scheduled preemption check at ", next_preemption_time)

        if preemption_event:
            if self.current_preemption_event and self.current_preemption_event in self.event_queue:
                if self.current_preemption_event.time > preemption_event.time:
                    self.event_queue.remove(self.current_preemption_event)
                    heapq.heapify(self.event_queue)
                else:
                    return 
            self.current_preemption_event = preemption_event
            heapq.heappush(self.event_queue, self.current_preemption_event)
        else:
            print("I found no preemption candidates", self.current_job.job_class.b, self.current_job.priority)
            
        
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
