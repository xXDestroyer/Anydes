"""Semantic relations -> local requirements -> existing executable compiler."""
from pathlib import Path
from .semantic_proposal import catalog, validate_semantic
from .general_reasoning import GeneralTaskPlan
from .operation_semantics import canonical_descriptor
from .source_names import canonical_source_name

class PlanLoweringError(ValueError):
    pass

class ExecutablePlanLowerer:
    def __init__(self, registry, workspace, metadata=None, external_artifacts=None):
        self.registry = registry
        self.workspace = Path(workspace).resolve(strict=True)
        self.metadata = catalog()["contracts"] if metadata is None else metadata
        self.external_artifacts = dict(external_artifacts or {})

    def select(self, kind, inputs, output):
        matches = []
        for row in self.metadata:
            if row["operation_kind"] != kind: continue
            if not self.registry.find_by_capability(row["capability"]): continue
            tool = self.registry.find_by_capability(row["capability"])[0]
            if self.registry.execution_denial(tool.manifest.tool_id, row["capability"]): continue
            if any(not self.registry.find_by_capability(cap) for cap in row.get("requires", [])): continue
            if output and (output["artifact_type"] != row["output_type"] or output["format"] not in row["formats"]): continue
            if any(v["artifact_type"] not in row["inputs"] for v in inputs): continue
            if row.get("input_formats") and any(v["format"] not in row["input_formats"] for v in inputs): continue
            if row["capability"] not in matches: matches.append(row["capability"])
        if not matches: raise PlanLoweringError("UNSUPPORTED_OPERATION:" + kind)
        if len(matches) != 1: raise PlanLoweringError("PLAN_AMBIGUOUS:capability_selection:" + kind)
        return matches[0]

    def lower_requirements(self, proposal):
        data = validate_semantic(proposal.to_dict())
        known, producers, steps, declared = {}, {}, [], []
        for name, ref in self.external_artifacts.items():
            known[name] = {'artifact_type':ref.artifact_type, 'format':ref.format, 'filename':ref.display_name}
            producers[name] = 'INPUT'
        content_specs, code_specs = [], []
        generated_code_specs = {}
        modification_spec = None
        ids = set()
        last = None
        for op in data["operations"]:
            oid, kind = op["operation_id"], op["operation_kind"]
            if kind == 'EDIT_ARTIFACT' and (not self.external_artifacts or
                    any(out['artifact_type'] != 'CODE' for out in op['outputs'])):
                raise PlanLoweringError('UNSUPPORTED_OPERATION:edit_requires_bound_code')
            if oid in ids: raise PlanLoweringError("PLAN_AMBIGUOUS:duplicate_operation")
            ids.add(oid)
            expected = {"CREATE_DOCUMENT":{"CONTENT"}, "CONVERT_DOCUMENT":{"DOCUMENT"},
                "TEST_CODE":{"CODE"}, "EDIT_ARTIFACT":{"CODE"}, "CREATE_ARCHIVE":{"CODE","DOCUMENT"}}.get(kind, set())
            names = list(op["inputs"])
            if not names and expected:
                candidates = [name for name,v in known.items() if v["artifact_type"] in expected]
                if len(candidates) != 1:
                    raise PlanLoweringError("PLAN_AMBIGUOUS:input_binding:" + oid)
                names = candidates
            if len(names) != len(set(names)) or any(name not in known for name in names):
                raise PlanLoweringError("PLAN_LOWERING_FAILED:dependency_not_preceding:" + oid)
            inputs = [known[name] for name in names]
            if expected and any(v["artifact_type"] not in expected for v in inputs):
                raise PlanLoweringError("PLAN_LOWERING_FAILED:input_type:" + oid)
            if kind != "CREATE_ARCHIVE" and expected and len(inputs) != 1:
                raise PlanLoweringError("PLAN_AMBIGUOUS:multiple_inputs:" + oid)
            outputs = op["outputs"]
            if len(outputs) != (0 if kind == "TEST_CODE" else 1):
                raise PlanLoweringError("PLAN_LOWERING_FAILED:output_cardinality:" + oid)
            out = outputs[0] if outputs else None
            if out and out["name"] in known: raise PlanLoweringError("PLAN_AMBIGUOUS:duplicate_output")
            spec = dict(op["specification"])
            if kind == "GENERATE_CONTENT":
                if names or out["artifact_type"] != "CONTENT" or out["format"] != "text":
                    raise PlanLoweringError("PLAN_LOWERING_FAILED:content_type")
                content_specs.append(spec)
                if len(content_specs) > 1: raise PlanLoweringError("UNSUPPORTED_OPERATION:multiple_content_generation")
            else:
                cap = self.select(kind, inputs, out)
                args = {}
                if kind == 'TEST_CODE':
                    args = {key:spec[key] for key in ('stdin','timeout','language') if key in spec}
                    producer_spec = generated_code_specs.get(names[0])
                    if producer_spec is not None and not producer_spec.get('test_cases'):
                        # A test operation's declared input is already semantic evidence.
                        # It must not be lost merely because the producer omitted cases.
                        cases = spec.get('test_cases')
                        if cases is None and 'stdin' in spec:
                            cases = [{'stdin':spec['stdin']}]
                        if cases is not None:
                            producer_spec['test_cases'] = cases
                if out:
                    filename = out.get("filename") or (out["name"] + "." + out["format"])
                    if out['artifact_type'] == 'CODE':
                        try:
                            filename = canonical_source_name(filename, out['format'])
                        except ValueError as exc:
                            raise PlanLoweringError('PLAN_LOWERING_FAILED:'+str(exc)) from exc
                    if kind == 'EDIT_ARTIFACT':
                        filename = inputs[0]['filename']
                    if Path(filename).name != filename or any(c in filename for c in '/\\:<>|?*') or filename in {".",".."}:
                        raise PlanLoweringError("PLAN_LOWERING_FAILED:filename")
                    if not Path(filename).suffix: filename += "." + out["format"]
                    if Path(filename).suffix.lower() != "." + out["format"]:
                        raise PlanLoweringError("PLAN_LOWERING_FAILED:format_mismatch")
                    args["destination" if kind == "CREATE_ARCHIVE" else "path"] = str(self.workspace / filename)
                if kind == "GENERATE_CODE":
                    if names: raise PlanLoweringError("PLAN_LOWERING_FAILED:unexpected_code_input")
                    spec["target_filename"] = filename
                    if 'test_cases' in spec and 'tests' in spec:
                        raise PlanLoweringError('semantic_test_invalid:duplicate_case_sources')
                    if 'tests' in spec:
                        spec['test_cases'] = spec.pop('tests')
                    spec.setdefault("language", {"py":"python","cpp":"cpp"}[out["format"]])
                    if spec.get("operation_descriptor"):
                        spec["operation_descriptor"] = canonical_descriptor(spec["operation_descriptor"])
                    code_specs.append(spec)
                    generated_code_specs[out['name']] = spec
                    if len(code_specs) > 1: raise PlanLoweringError("UNSUPPORTED_OPERATION:multiple_code_generation")
                if kind == 'EDIT_ARTIFACT':
                    if modification_spec is not None or code_specs:
                        raise PlanLoweringError('PLAN_AMBIGUOUS:multiple_code_edit')
                    ref = self.external_artifacts.get(names[0])
                    if ref is None:
                        raise PlanLoweringError('CONTEXT_ARTIFACT_BINDING_REQUIRED')
                    modification_spec = {**spec, 'target_artifact':ref.display_name,
                        'language':'python' if ref.format == 'py' else 'cpp'}
                    steps.append({'step_id':oid+'_read', 'capability':'developer.read_code',
                        'purpose':'READ_BOUND_ARTIFACT', 'arguments':{'path':ref.path},
                        'consumes_artifacts':names, 'produces_artifacts':[]})
                if kind == "CREATE_DOCUMENT" and not content_specs:
                    raise PlanLoweringError("PLAN_LOWERING_FAILED:content_spec_missing")
                step = {"step_id":oid,"capability":cap,"purpose":kind,"arguments":args,
                    "depends_on":list(dict.fromkeys([producers[n] for n in names if known[n]["artifact_type"] != "CONTENT" and producers[n] != 'INPUT'] + ([last] if last else []))),
                    "produces_artifacts":[out["name"]] if out else [],
                    "consumes_artifacts":[n for n in names if known[n]["artifact_type"] != "CONTENT"],
                    "consumes_values":[n for n in names if known[n]["artifact_type"] == "CONTENT"]}
                steps.append(step)
                if kind == 'TEST_CODE' and 'execution_purpose' in spec:
                    from .execution_contract import ExecutionPurpose
                    purpose = ExecutionPurpose(spec['execution_purpose'])
                    step['purpose'] = 'EXECUTE_FOR_VERIFICATION' if purpose == ExecutionPurpose.VERIFICATION else 'EXECUTE_AS_USER_ACTION'
                last = oid
                if out: declared.append({"binding":out["name"],"artifact_type":out["artifact_type"],"format":out["format"]})
            if out:
                known[out["name"]] = out
                producers[out["name"]] = oid
        execution_only = all(op['operation_kind'] == 'TEST_CODE' for op in data['operations'])
        if (not data["requested_final_outputs"] and not execution_only) or any(n not in known or known[n]["artifact_type"] == "CONTENT" or producers[n] == 'INPUT' for n in data["requested_final_outputs"]):
            raise PlanLoweringError("PLAN_LOWERING_FAILED:final_output:requested=" + ','.join(data['requested_final_outputs']) + ':available=' + ','.join(known))
        if content_specs and code_specs: raise PlanLoweringError("UNSUPPORTED_OPERATION:mixed_generation")
        task_type = 'MODIFY_CODE' if modification_spec else "GENERATE_CODE" if code_specs else "GENERATE_DOCUMENT" if content_specs else "TOOL_COMPOSITION"
        artifacts = {"semantic_lowered":True}
        if content_specs:
            first = next((op["outputs"][0] for op in data["operations"] if op["operation_kind"] == "CREATE_DOCUMENT"), None)
            if not first: raise PlanLoweringError("PLAN_LOWERING_FAILED:document_writer_missing")
            artifacts.update(format=first["format"],filename=Path(steps[0]["arguments"]["path"]).name)
        inputs = {'code_modification_spec':modification_spec} if modification_spec else {"code_spec":code_specs[0]} if code_specs else {"content_spec":content_specs[0]} if content_specs else {}
        return GeneralTaskPlan.from_dict({"task_id":proposal.task_id,"goal":data["goal"],"task_type":task_type,
            "inputs":inputs,"steps":steps,"required_capabilities":list(dict.fromkeys(s["capability"] for s in steps)),
            "desired_outputs":declared,"requested_final_outputs":[r for r in declared if r["binding"] in data["requested_final_outputs"]],
            "artifact_requirements":artifacts,"constraints":data.get("constraints",[]),
            "needs_tools":True,"needs_content_generation":bool(content_specs or code_specs or modification_spec),"risk":"low"})

    def lower(self, proposal, action_builder, adapter, source_request):
        requirements = self.lower_requirements(proposal)
        action = action_builder(requirements, source_request)
        if action is None: raise PlanLoweringError("PLAN_LOWERING_FAILED:not_executable")
        return adapter.compile(action, source="GENERAL_REASONER", task_id=proposal.task_id)
