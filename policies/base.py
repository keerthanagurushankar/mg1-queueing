class Policy:
    def __init__(self, policy_name, priority_fn=None, is_preemptive=False,
                 is_dynamic_priority=False, calculate_overtake_time=lambda j1, j2, t: None):
        self.policy_name = policy_name
        self.priority_fn = priority_fn
        self.is_preemptive = is_preemptive
        self.is_dynamic_priority = is_dynamic_priority
        self.calculate_overtake_time = calculate_overtake_time

# SIMPLE POLICIES
FCFS = Policy("FCFS", priority_fn=lambda r, s, t, k: t, is_preemptive=False)
SRPT = Policy("SRPT", priority_fn=lambda r, s, t, k: -r, is_preemptive=True, is_dynamic_priority=True)

# CLASS BASED POLICIES
NPPrio12 = Policy("NPPrio12", priority_fn=lambda r, s, t, k: -k, is_preemptive=False)
PPrio12 = Policy("PPrio12", priority_fn=lambda r, s, t, k: -k, is_preemptive=True)
NPPrio21 = Policy("NPPrio21", priority_fn=lambda r, s, t, k: k, is_preemptive=False)
PPrio21 = Policy("PPrio21", priority_fn=lambda r, s, t, k: k, is_preemptive=True)
