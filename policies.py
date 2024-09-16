class Policy:
    def __init__(self, policy_name, priority_fn=None, is_preemptive=False,
                 is_dynamic_priority=False):
        self.policy_name = policy_name
        self.priority_fn = priority_fn
        self.is_preemptive = is_preemptive
        self.is_dynamic_priority = is_dynamic_priority 

FCFS = Policy("FCFS")
SRPT = Policy("SRPT", priority_fn=lambda r, s, t, k:-r, is_preemptive=True, is_dynamic_priority = True)
NPPrio12 = Policy("NPPrio12", priority_fn=lambda r, s, t, k:-k, is_preemptive=False)
PPrio12 = Policy("PPrio12", priority_fn=lambda r, s, t, k:-k, is_preemptive=True)

def AccPrio(b1, b2, is_preemptive=False):
    def V(r, s, t, k):
        return (k == 1) * b1 * t + (k == 2) * b2 * t
    return Policy("NPAccPrio", priority_fn=V, is_preemptive=is_preemptive,
                  is_dynamic_priority=True)
