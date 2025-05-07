import heapq
from collections import deque
import random, math
import logging, os, json
import lib
logging.basicConfig(level=logging.INFO)

class Event:
    def __init__(self, event_type, time, job=None):
        self.event_type = event_type
        self.time = time
        self.job = job

    def __lt__(self, other):
        return self.time < other.time

class JobClass:
    def __init__(self, index, l, S, path=None): 
        self.index = index
        self.l = l
        self.S = S
        self.priority = None # will be initialized by MG1 class
        
        self.arrival_sequence = None
        if path:
            with open(f'{path}/arrival_sequence{index}.json', 'r') as f:
                self.arrival_sequence = json.load(f)
                self.sequence_index = -1

        self.job_queue = deque([])
        logging.debug(f"I initialized job_queue of {index} as {self.job_queue}")

    def generate_next_job(self, current_time):
        if self.arrival_sequence and self.sequence_index < len(self.arrival_sequence)-1:
            self.sequence_index += 1
            return Job(self.arrival_sequence[self.sequence_index]['arrival_time'],
                       self.arrival_sequence[self.sequence_index]['job_size'], self)

        return Job(current_time + random.expovariate(self.l), self.S(), self)

class Job:
    def __init__(self, arrival_time, service_time, job_class):
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.remaining_time = service_time        
        self.job_class = job_class
        self.priority = job_class.priority(self.remaining_time, self.service_time, 0)

    def __lt__(self, other):
        if self.priority == other.priority: 
            return self.arrival_time < other.arrival_time # older is better if unspecified
        return self.priority > other.priority # higher priority is better    

class MG1:
    def __init__(self, job_classes, policy,
                 inspection_rate=1, simulation_time=10**5):
        # Parameters
        self.job_classes = job_classes
        self.policy = policy
        
        self.simulation_time = 10**5 #simulation_time
        self.inspection_rate = 1 #inspection_rate
        
        # initialize priority_function of job classes
        for k, job_class in enumerate(job_classes):
            job_class.priority = lambda r, s, t, idx=job_class.index:\
              policy.priority_fn(r, s, t, idx)

        # System state
        self.event_queue = []
        self.current_time = 0
        self.current_job = None # if busy, job being served
        self.current_service_start_time = None # if busy, last service start time
        
        self.num_inspections_total = 0 
        self.num_inspections_failed = 0

        # Metrics
        self.metrics = [] # stores records for each job which departs system

    def initialize(self):     
        for job_class in self.job_classes:
            job = job_class.generate_next_job(0)
            heapq.heappush(self.event_queue, Event('Arrival', job.arrival_time, job))
            job_class.job_queue = deque([]) # Needed if same job class defs are reused       
            logging.debug(f"Inititalizing job queue {job_class.index, len(job_class.job_queue)}")

        heapq.heappush(self.event_queue, Event('Inspection', random.expovariate(self.inspection_rate)))

    def run(self):
        self.initialize()

        while self.current_time < self.simulation_time:
            assert self.event_queue, "Should have remaining events scheduled"

            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            logging.debug(f"EVENT: {event.time, event.event_type}, ")
                        #  f"job index: {event.job.job_class.index if event.job else None}")

            if event.event_type == 'Arrival':
                self.handle_arrival(event)
            elif event.event_type == 'Departure':                  
                self.handle_departure(event)
            elif event.event_type == 'Inspection':
                self.handle_inspection()
            elif event.event_type == 'PreemptionCheck':
                self.handle_preemption_check(event)

        print(f"Failed {self.num_inspections_failed} / {self.num_inspections_total} inspections")

    def handle_arrival(self, event):
        # Schedule next arrival for the appropriate job_class
        next_job = event.job.job_class.generate_next_job(self.current_time)
        heapq.heappush(self.event_queue, Event('Arrival', next_job.arrival_time, next_job))

        # if idle:
        #    start serving me
        # elif preemptive:
        #    if my job queue already had stuff [or current_job in service is my class]:
        #       just enqueue me
        #    else I may preempt right away or need a preemption check:
        #       if my current priority is higher than current_job's current priority:
        #          preempt current job and serve me
        #          [if dynamic: schedule preemption checks for everyone else]
        #       elif dynamic:
        #          if we're equal-ish, check who I should actually be serving
        #             it's likely that i'm just about to overtake the current job
        #             if so, preempt current job and serve me, schedule checks for all
        #          if my current priority is lower than current job:
        #             enqueue me
        #             schedule a preemption check for me in the future
        #       else:
        #          just enqueue me
        # else: [non preemptive and busy]
        #    just enqueue me

        new_job = event.job
        
        if self.current_job is None:
            self.start_service(new_job)
            
        elif self.policy.is_preemptive:
            logging.debug("System was busy, I may preempt right away or need future check")
            if new_job.job_class.job_queue or self.current_job.job_class == new_job.job_class:
                new_job.job_class.job_queue.append(new_job)
            else:
                # I may preempt right away or need a preemption check
                self.current_job.priority = self.current_job.job_class.priority(
                    self.current_job.remaining_time, self.current_job.service_time,
                    self.current_time - self.current_job.arrival_time)
                
                if new_job.priority > self.current_job.priority:
                    self.preempt_current_job()
                    self.start_service(new_job)
                        
                elif self.policy.is_dynamic_priority:
                    
                    if False: #math.isclose(new_job.priority, self.current_job.priority, rel_tol=1e-3):
                        
                        future_delta = 1e-3
                        current_job_future_prio = self.current_job.job_class.priority(
                          self.current_job.remaining_time, self.current_job.service_time,
                          self.current_time + future_delta - self.current_job.arrival_time)
                        new_job_future_prio = new_job.job_class.priority(
                            new_job.remaining_time, new_job.service_time, future_delta)

                        if new_job_future_prio > current_job_future_prio:
                            self.preempt_current_job()
                            self.start_service(new_job)

                    else:
                        new_job.job_class.job_queue.append(new_job)
                        self.schedule_preemption_check(new_job)
                
                else:
                    new_job.job_class.job_queue.append(new_job)                    
        
        else: # [non preemptive and busy]
            new_job.job_class.job_queue.append(new_job)


    def handle_departure(self, event):
        if self.current_job != event.job or self.current_time != \
           self.current_job.remaining_time + self.current_service_start_time:
            assert self.policy.is_preemptive, "Stale departures must only occur with preemption"
            logging.debug("Ignoring stale departure event")
            return 
        
        self.record_metrics(event.job, self.current_time)
        self.current_job, self.current_service_start_time = None, None

        if self.get_updated_top_priorities()[0]:
            self.start_service()

    def schedule_preemption_check(self, new_job = None):
        assert self.policy.is_dynamic_priority and self.policy.is_preemptive
        assert self.current_job is not None

        preemption_event = None
        
        if new_job:
            # if a diff class just arrived, it may preempt the curr job at future time
            t_overtake = self.policy.calculate_overtake_time(self.current_job,
                                                             new_job, self.current_time)
            if t_overtake: 
                preemption_event = Event('PreemptionCheck', t_overtake, (self.current_job, new_job))
        else:
            # when starting new service, check if any job in queue may grow to overtake priority
            min_overtake_time, overtake_job = float('inf'), None
            for job in self.get_updated_top_priorities()[0]:
                t_overtake = self.policy.calculate_overtake_time(self.current_job, job,
                                                                 self.current_time)
                if t_overtake and t_overtake < min_overtake_time:
                    min_overtake_time, overtake_job = t_overtake, job
                    
            if overtake_job:
                preemption_event = Event('PreemptionCheck', min_overtake_time,
                                         (self.current_job, overtake_job))

        if preemption_event:
            logging.debug(f"I'm scheduling preemption check at {preemption_event.time}")
            heapq.heappush(self.event_queue, preemption_event)

    def handle_preemption_check(self, event):        
        event_current_job, overtake_job = event.job
        if self.current_job != event_current_job:
            logging.debug("Ignoring stale preemption check")
            return

        self.current_job.priority = self.current_job.job_class.priority(
                    self.current_job.remaining_time, self.current_job.service_time,
                    self.current_time - self.current_job.arrival_time)
        overtake_job.priority = overtake_job.job_class.priority(
            overtake_job.remaining_time, overtake_job.service_time,
            self.current_time - overtake_job.arrival_time)

        if overtake_job.priority >= self.current_job.priority:
            self.preempt_current_job()
            self.start_service(overtake_job)

    def preempt_current_job(self):
        # update completed service of current_job & put it back in its queue [front of its queue]        
        assert self.current_job is not None
        
        time_in_service = self.current_time - self.current_service_start_time
        self.current_job.remaining_time = self.current_job.remaining_time - time_in_service
        
        self.current_job.job_class.job_queue.appendleft(self.current_job)
        self.current_job, self.current_service_start_time = None, None
    
    def start_service(self, job=None):
        # if given job, assert noone in service rn and start serving me
        # if not given job, update prios and pick highest priority job
        # if dynamic, schedule preemption checks

        assert self.current_job is None # also assert job or non empty queue
        self.current_service_start_time = self.current_time
        
        if job:
            self.current_job = job
        else:
            top_job_class = self.get_updated_top_priorities()[1]
            self.current_job = top_job_class.job_queue.popleft()

        assert self.current_time >= self.current_job.arrival_time, \
            "Tried to start service before arrival"
        logging.debug(f"I'm starting work on class {self.current_job.job_class.index} "
                      f"with priority {self.current_job.priority} "
                      f"and work {self.current_job.remaining_time} at {self.current_time}")
        
        departure_time = self.current_time + self.current_job.remaining_time
        departure_event = Event('Departure', departure_time, self.current_job)
        heapq.heappush(self.event_queue, departure_event)

        if self.policy.is_preemptive and self.policy.is_dynamic_priority:
            self.schedule_preemption_check()

    def get_updated_top_priorities(self):
        top_waiting_jobs = []
        top_job_class, top_prio = None, -float('inf')
        
        for job_class in self.job_classes:
            if job_class.job_queue:
                job = job_class.job_queue[0]
                job.priority = job.job_class.priority(job.remaining_time,
                        job.service_time, self.current_time - job.arrival_time)

                if job.priority > top_prio:
                    top_job_class, top_prio = job_class, job.priority
                top_waiting_jobs.append(job)

                logging.debug(f"Top waiting job in {job_class.index} has prio {job.priority} "
                              f"num waiting jobs is {len(job_class.job_queue)}")

        logging.debug(f"{len(top_waiting_jobs)} classes with waiting jobs, "
                      f"top job class is {top_job_class.index if top_job_class else None}")
        
        return top_waiting_jobs, top_job_class
    
    def handle_inspection(self):
        self.num_inspections_total += 1
        next_inspection_time = self.current_time + random.expovariate(self.inspection_rate)
        heapq.heappush(self.event_queue, Event('Inspection', next_inspection_time))        
        
        if self.current_job is None:
            assert self.current_service_start_time is None
            
            for job_class in self.job_classes:
                assert not job_class.job_queue, "No waiting jobs while idle"
        else:
            assert self.current_service_start_time is not None
            
            if self.policy.is_preemptive:
                current_job_prio = self.current_job.job_class.priority(
                    self.current_job.remaining_time, self.current_job.service_time,
                    self.current_time - self.current_job.arrival_time)
                
                for job_class in self.job_classes:
                    if job_class.job_queue:
                        waiting_job = job_class.job_queue[0]
                        waiting_prio = waiting_job.job_class.priority(
                          waiting_job.remaining_time, waiting_job.service_time,
                          self.current_time - waiting_job.arrival_time)

                        if waiting_prio > current_job_prio and \
                           not math.isclose(waiting_prio, current_job_prio, rel_tol=0.01):
                            logging.debug(f"Inspection fail: waiting job of class {job_class.index} "
                                f"with priority {waiting_prio} while serving "
                                f"{self.current_job.job_class.index} with prio {current_job_prio}")
                            self.num_inspections_failed += 1

                            t_overtake = self.policy.calculate_overtake_time(self.current_job,
                                                             waiting_job, self.current_time)
                            logging.debug(f"Current time {self.current_time}, I would schedule "
                                         f"overtake at {t_overtake}")

                            

            else:
                # if non-preemptive, at current_service_start_time,
                # every job in sys had lower prio than current_job
                current_job_prio = self.current_job.job_class.priority(
                    self.current_job.remaining_time, self.current_job.service_time,
                    self.current_service_start_time - self.current_job.arrival_time)
                
                for job_class in self.job_classes:
                    if job_class.job_queue:
                        waiting_job = job_class.job_queue[0]
                        
                        if waiting_job.arrival_time <= self.current_service_start_time:
                            waiting_prio = waiting_job.job_class.priority(
                              waiting_job.remaining_time, waiting_job.service_time,
                              self.current_service_start_time - waiting_job.arrival_time)

                            if waiting_prio > current_job_prio:
                                logging.debug("Inspection fail")
                                self.num_inspections_failed += 1

    def record_metrics(self, job, departure_time):
        job_metrics = {'job_class': job.job_class.index, 
                       'arrival_time': job.arrival_time,
                       'job_size': job.service_time,
                       'departure_time': departure_time,                       
                       'response_time': departure_time - job.arrival_time,
                       'waiting_time': departure_time - job.arrival_time
                           - job.service_time,
                       'priority': job.priority} # at completion
        # [I don't think this priority gets updated though]
        self.metrics.append(job_metrics)

    def save_metrics(self, path):
        os.makedirs(path, exist_ok=True)
        json.dump(self.metrics, open(os.path.join(path, "metrics.json"), 'w'))
            
        for job_class in self.job_classes:
            arrival_sequence = [job for job in self.metrics
                                if job['job_class'] == job_class.index]
            fname = os.path.join(path, f"arrival_sequence{job_class.index}.json")
            json.dump(arrival_sequence, open(fname, 'w'))
        
        logging.info(f"Saved metrics to {path}")
