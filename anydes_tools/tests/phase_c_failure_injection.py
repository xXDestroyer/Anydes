"""Explicit replay-only failure injection; never imported by the human bootstrap."""
from pathlib import Path
import sys


class FailCodeVerificationOnce:
    def __init__(self, controller, observer):
        self.controller = controller
        self.observer = observer
        self.injected = 0
        self.original = None

    def install(self):
        entry = Path(getattr(sys.modules.get("__main__"), "__file__", "")).resolve()
        expected = Path(__file__).with_name("run_full_runtime_phase_a.py").resolve()
        if entry != expected:
            raise RuntimeError("failure_injection_requires_explicit_test_harness")
        import anydes_tools.conversation.controller as runtime
        self.original = runtime.verify_code_test

        def verify(stdout, test):
            passed = self.original(stdout, test)
            pending = self.controller.state.pending_task
            if passed and not self.injected and pending.plan is not None:
                self.injected += 1
                self.observer.emit("test_failure_injection",
                    error_code="TEST_FORCED_VERIFY_FAILURE", stage="VERIFY",
                    task_id=pending.task_id, plan_revision=pending.plan_revision,
                    step_id=self.controller.last_step_results[-1].step_id,
                    test_id=test.test_id, real_verification_passed=True,
                    fail_once=True, final_commit_allowed=False)
                return False
            return passed

        self.wrapper = verify
        runtime.verify_code_test = verify
        self.observer.emit("test_failure_hook", enabled=True, fail_once=True)

    def close(self):
        if self.original is None:
            return
        import anydes_tools.conversation.controller as runtime
        runtime.verify_code_test = self.original
        self.observer.emit("test_failure_hook", enabled=False, injected=self.injected,
            original_verifier_restored=runtime.verify_code_test is self.original)

