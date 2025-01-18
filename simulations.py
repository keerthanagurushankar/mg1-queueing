import heapq
import json, os
import logging
import random, math
import lib

logging.basicConfig(level=logging.INFO)

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
        if self.priority == other.priority:
            return self.arrival_time < other.arrival_time # older is better if unspecified
        return self.priority > other.priority # higher priority is better

    def current_priority(self, current_time):
        return self.job_class.priority(self.remaining_time, self.service_time,
                                        current_time - self.arrival_time)

class JobClass:
    def __init__(self, index, l, S, path=None):
        self.index = index
        self.l = l
        self.S = S
        self.priority = None

        self.arrival_sequence = None
        if path:
            with open(f'{path}/arrival_sequence{index}.json', 'r') as f:
                self.arrival_sequence = json.load(f)
                self.sequence_index = -1

    def generate_next_job(self, current_time):
        if self.arrival_sequence and self.sequence_index < len(self.arrival_sequence)-1:
            self.sequence_index += 1
            return Job(self.arrival_sequence[self.sequence_index]['arrival_time'],
                       self.arrival_sequence[self.sequence_index]['job_size'], self)

        return Job(current_time + random.expovariate(self.l), self.S(), self)

class MG1:
    def __init__(self, job_classes, policy):
        # Parameters
        self.job_classes = job_classes
        self.policy = policy

        self.simulation_time = 10**5
        self.inspection_rate = 1

        # initialize priority_function of job classes
        for k, job_class in enumerate(job_classes):
            if policy.priority_fn is None:
                job_class.priority = lambda r, s, t, idx=job_class.index: -idx
            elif isinstance(policy.priority_fn, list):
                job_class.priority = policy.priority_fn[k]
            else:
                job_class.priority = lambda r, s, t, idx=job_class.index: policy.priority_fn(r, s, t, idx)

        # System state
        self.event_queue = [] # holds events in order of event time
        self.job_queue = [] # holds waiting jobs in order of priority
        self.current_time = 0
        self.current_job = None # if busy, is job being served
        self.current_service_start_time = None # if busy, is time of start of last service
        self.current_departure_event = None # departure event of job being served
        self.current_preemption_event = None # the preemption check to occur nearest in future

        # Metrics
        self.metrics = []

    def initialize(self):
        for job_class in self.job_classes:
            job = job_class.generate_next_job(0)
            heapq.heappush(self.event_queue, Event('Arrival', job.arrival_time, job))

        #heapq.heappush(self.event_queue, Event('Inspection', random.expovariate(self.inspection_rate)))

    def run(self):
        self.initialize()

        while self.event_queue and self.current_time < self.simulation_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            logging.debug(f"EVENT: {event.time, event.event_type, event.job.job_class.index if event.job else None}")

            if event.event_type == 'Arrival':
                self.handle_arrival(event)
            elif event.event_type == 'Departure':
                self.handle_departure(event)
            elif event.event_type == 'Inspection':
                self.handle_inspection()
            elif event.event_type == 'PreemptionCheck':
                self.handle_preemption_check()

    def handle_arrival(self, event):
        job = event.job
        heapq.heappush(self.job_queue, job)

        # Schedule next arrival for the appropriate job_class
        next_job = job.job_class.generate_next_job(self.current_time)
        heapq.heappush(self.event_queue, Event('Arrival', next_job.arrival_time, next_job))

        if self.current_job is None:
            self.start_service()
        elif self.policy.is_preemptive:
            self.update_priorities() # if dynamic
            if self.current_job.priority < job.priority:
                # the arrival has higher priority so we should preempt service
                logging.debug("Arrival is more important than current job")
                self.preempt_current_job()
            else:
                # arrival is low prio now but may become higher priority later (if dynamic)
                self.schedule_preemption_check(job)

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
                    assert job_priority <= current_job_priority + 0.1, "Not working on highest prio job" + f"\
                      {current_job_priority, job_priority, self.current_job.job_class.index, job.job_class.index}"
            else:
                for job in self.job_queue:
                    assert job.priority <= self.current_job.priority or job.arrival_time \
                       >= self.current_service_start_time, "Not working on highest prio job"

        next_inspection_time = self.current_time + random.expovariate(self.inspection_rate)
        heapq.heappush(self.event_queue, Event('Inspection', next_inspection_time))

    def handle_preemption_check(self):
        # may have to update remaining service time of current job here for srpt
        self.current_preemption_event = None
        self.preempt_current_job()
        #self.update_priorities()

        #if self.job_queue and self.job_queue[0].priority > self.current_job.priority:
         #   self.preempt_current_job()

    def start_service(self):
        # assume system is not empty and job priorities are up to date
        # start working on job of current highest priority
        self.current_job = heapq.heappop(self.job_queue)
        self.current_service_start_time = self.current_time

        departure_time = self.current_time + self.current_job.remaining_time
        assert self.current_time >= self.current_job.arrival_time, "Tried to start service before arrival"

        self.current_departure_event = Event('Departure', departure_time, self.current_job)
        heapq.heappush(self.event_queue, self.current_departure_event)

        logging.debug(f"I'm starting work on class f{self.current_job.job_class.index}"
                      f"with priority {self.current_job.priority} at {self.current_time}")
        self.schedule_preemption_check() # if dynamic and preemptive

    def update_priorities(self):
        if not self.policy.is_dynamic_priority:
            return

        updated_priority = False
        for job in self.job_queue:
            new_priority = job.current_priority(self.current_time)
            if not math.isclose(new_priority, job.priority):
                logging.debug(f"I'm updating priority of {job.job_class.index, job.arrival_time}"
                              f"from {job.priority} to {new_priority}")
                updated_priority = True
                job.priority = new_priority

        if self.current_job:
            self.current_job.priority = self.current_job.current_priority(self.current_time)

        if updated_priority:
            heapq.heapify(self.job_queue)

    def preempt_current_job(self):
        # if busy, put currently served job in queue,
        # update and start work on whatever is the new highest priority job
        if self.current_job is None:
            return

        time_in_service = self.current_time - self.current_service_start_time
        self.current_job.remaining_time = self.current_job.remaining_time - time_in_service
        heapq.heappush(self.job_queue, self.current_job)

        self.event_queue.remove(self.current_departure_event)
        heapq.heapify(self.event_queue)

        self.update_priorities()
        self.start_service()

    def schedule_preemption_check(self, new_job = None):
        # check if new arrival or some job in queue may preempt the current service
        if not self.policy.is_dynamic_priority or not self.policy.is_preemptive:
            return

        preemption_event = None

        # calculate what the system currently considers the earliest preemption into preemption_event
        if new_job:
            # if a higher b job just arrived, it may preempt the low b job if it hasn't completed service
            t_overtake = self.policy.calculate_overtake_time(self.current_job, new_job, self.current_time)
            if t_overtake: # and t_overtake < self.current_departure_event.time:
                preemption_event = Event('PreemptionCheck', t_overtake, new_job)
        else:
            # when starting new service, check and schedule if any job in queue may grow to overtake priority
            min_overtake_time, overtake_job = float('inf'), None
            for job in self.job_queue:
                t_overtake = self.policy.calculate_overtake_time(self.current_job, job, self.current_time)
                if t_overtake and t_overtake < min_overtake_time:
                    min_overtake_time, overtake_job = t_overtake, job

            if overtake_job:
                preemption_event = Event('PreemptionCheck', min_overtake_time, overtake_job)

        # if found a preemption and it's better than current_preemption_event, update current_preemption_event
        if preemption_event:
            if self.current_preemption_event and self.current_preemption_event in self.event_queue:
                if self.current_preemption_event.time > preemption_event.time: # + 0.1 \
                   #and self.current_preemption_event.job != preemption_event.job:
                    logging.debug("I'm swapping preemption event")
                    self.event_queue.remove(self.current_preemption_event)
                    heapq.heapify(self.event_queue)

                    self.current_preemption_event = preemption_event
                    heapq.heappush(self.event_queue, self.current_preemption_event)
            else:
                self.current_preemption_event = preemption_event
                heapq.heappush(self.event_queue, self.current_preemption_event)

    def record_metrics(self, job, departure_time):
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
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metrics.json"), 'w') as metrics_file:
            json.dump(self.metrics, metrics_file)

        for job_class in self.job_classes:
            arrival_sequence = [job for job in self.metrics
                                if job['job_class'] == job_class.index]
            with open(os.path.join(path, f"arrival_sequence{job_class.index}.json"), 'w') as f:
                json.dump(arrival_sequence, f)

        logging.info(f"Saved metrics to {path}")
