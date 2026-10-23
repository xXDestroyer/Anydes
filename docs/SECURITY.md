# Security Model

Anydes uses default-deny policy at tool, permission, path and publication boundaries.
Model text is a proposal, not operational authority. The cognitive Action Manager
and tool execution contracts remain distinct layers.

- Tool manifests declare capabilities, schemas, effects and resource timeouts.
- PermissionBridge checks allowed scopes and required confirmation.
- FilesystemBoundary checks authorized paths, protected locations and reparse risks.
- ToolExecutor validates input/output envelopes and dispatches trusted workers.
- Generated code is not executed through an unrestricted fallback subprocess.

## Windows B2

The real `windows_sandbox.py` implementation uses Windows AppContainer/LPAC tokens,
private runtime copies, narrowly scoped filesystem ACLs and Job Objects. Network
capabilities are denied by default. Time, memory, process, CPU, output and filesystem
usage are bounded. Cleanup terminates descendants and removes private execution
directories/profiles. Unsupported isolation fails closed.

The accepted staging overlay fixes only a usage-accounting TOCTOU: an entry that
disappears between enumeration and `lstat()` is skipped. PermissionError and other
unexpected OSError cases still propagate; reparse/hardlink and quota checks remain.
This does not make Windows isolation portable to Linux by changing its name.

## Execution purpose

User-requested execution and internal verification execution are separate concepts.
Creating a source file does not automatically authorize running it as an action.
Internal verification may run in B2 when required by the validated artifact
contract. BASIC verification may legitimately require no execution. No artificial
run should be introduced merely for a demonstration checklist.

## Claims and data

Execution success is not independent correctness. TestSpec/oracle checks determine
what is verified; publication restricts claims to evidence. Artifact location can
be answered from authoritative metadata without a new execution result.

No private memory, credentials, microphone samples, voice reference recordings or
production policies with personal configuration are distributed. Example values
are placeholders. Review generated tools and configuration before granting any
permissions. This source release is not a security certification or a promise of
perfect containment.
