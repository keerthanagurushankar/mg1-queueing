class Policy:
    def __init__(self, policy_name, priority_fn=None, is_preemptive=False,
                 is_dynamic_priority=False):
        self.policy_name = policy_name
        self.priority_fn = priority_fn
        self.is_preemptive = is_preemptive
        self.is_dynamic_priority = is_dynamic_priority 

FCFS = Policy("FCFS")
SRPT = Policy("SRPT", priority_fn=lambda r, s, t, k:-r, is_preemptive=True, is_dynamic_priority = True)

V1, V2 = lambda r, s, t:-1, lambda r, s, t:-2
NPPrio12 = Policy("NPPrio12", priority_fn=[V1, V2], is_preemptive=False)
PPrio12 = Policy("PPrio12", priority_fn=[V1, V2], is_preemptive=True)

def AccPrio(b1, b2, is_preemptive=False):
    V1 = lambda r, s, t: b1 * t
    V2 = lambda r, s, t: b2 * t
    policy_name = "PAccPrio" if is_preemptive else "NPAccPrio"
    return Policy((policy_name, b1, b2), priority_fn=[V1, V2],
                  is_preemptive=is_preemptive,
                  is_dynamic_priority=True)

def Lookahead(alpha):
    V1 = lambda r, s, t: t
    V2 = lambda r, s, t: alpha
    return Policy(("Lookahead", alpha), priority_fn=[V1, V2],
                  is_preemptive=True, is_dynamic_priority=True)
