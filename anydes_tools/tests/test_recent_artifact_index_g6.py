"""UNIT index contracts; INTEGRATION prepared cognition with real executor/ZIP."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from anydes_tools.conversation.artifact_state import RecentArtifactIndex
from anydes_tools.conversation.dataflow import ArtifactBindings, ArtifactState, file_hash
from anydes_tools.conversation.planner import plan_from_text
from anydes_tools.tests import test_artifact_dataflow_phase_d as phase_d


class IndexContract(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.index = RecentArtifactIndex()

    def tearDown(self):
        self.temp.cleanup()

    def artifact(self, name='suma.py', kind='CODE', **metadata):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'known bytes')
        return ArtifactBindings('producer').produce('result', path, 'write',
            state=ArtifactState.COMMITTED, artifact_type=kind, metadata=metadata)

    def test_all_user_addressable_types(self):
        for kind, name in [('CODE','suma.py'), ('DOCUMENT','nota.docx'), ('PDF','nota.pdf'),
                           ('ARCHIVE','nota.zip'), ('TEXT','nota.txt'), ('OTHER','nota.bin')]:
            with self.subTest(kind=kind):
                ref = self.artifact(name, kind)
                record = self.index.register(ref)
                self.assertEqual(record.runtime_ref, ref)
                self.assertEqual(self.index.resolve(name, require_unique=True), ref.path)

    def test_nonpublic_states_and_metadata(self):
        ref = self.artifact(user_addressable=False)
        self.assertIsNone(self.index.register(ref))
        for state in (ArtifactState.TEMPORARY, ArtifactState.FAILED, ArtifactState.DELETED, ArtifactState.VERIFIED):
            self.assertIsNone(self.index.register(replace(ref, state=state, metadata={})))
        self.assertEqual(self.index.snapshot(), [])

    def test_contextual_program_and_exact_name(self):
        ref = self.artifact()
        self.index.register(ref)
        self.index.register(self.artifact('documento.docx', 'DOCUMENT'))
        self.assertEqual(self.index.resolve('comprime el programa recien creado', require_unique=True), ref.path)
        self.assertEqual(self.index.resolve('comprime suma.py', require_unique=True), ref.path)

    def test_duplicate_name_ambiguous(self):
        self.index.register(self.artifact())
        self.index.register(self.artifact('other/suma.py'))
        self.assertEqual(self.index.resolve('comprime suma.py', require_unique=True), '')
        self.assertEqual(self.index.resolve('', explicit_filename='suma.py'), '')

    def test_new_version_retains_identity(self):
        ref = self.artifact()
        self.index.register(ref)
        Path(ref.path).write_bytes(b'edited version')
        newer = replace(ref, version=2, hash=file_hash(ref.path))
        self.index.register(newer)
        self.index.register(ref)
        self.assertEqual(self.index.ref_by_path(ref.path), newer)
        self.assertEqual(len(self.index.artifacts), 1)

    def test_deleted_not_resolvable(self):
        ref = self.artifact()
        self.index.register(ref)
        self.index.remove(ref.path)
        self.assertEqual(self.index.resolve('suma.py'), '')
        self.assertIsNone(self.index.ref_by_path(ref.path))

    def test_stale_not_resolvable(self):
        ref = self.artifact()
        self.index.register(ref)
        Path(ref.path).write_bytes(b'changed externally')
        self.assertEqual(self.index.resolve('suma.py'), '')

    def test_missing_name_no_fallback(self):
        ref = self.artifact()
        self.index.register(ref)
        self.assertEqual(self.index.resolve('', explicit_filename='missing.py'), '')
        self.assertEqual(self.index.resolve('comprime missing.py', require_unique=True), '')
        self.assertEqual(self.index.resolve_from_workspace('suma.py', str(self.root)), '')


class ArchiveIntegration(unittest.TestCase):
    setUp = phase_d.DataflowIntegration.setUp
    tearDown = phase_d.DataflowIntegration.tearDown
    controller = phase_d.DataflowIntegration.controller

    def test_exact_ref_survives_adapter_and_physical_archive(self):
        c = self.controller()
        source = self.workspace / 'suma.py'
        source.write_bytes(b'print(1 + 2)\n')
        ref = ArtifactBindings('create').produce('code', source, 'write',
            state=ArtifactState.COMMITTED, artifact_type='CODE')
        c.state.recent_artifact_state.register(ref)
        # No operational-map fallback: the selected ref itself must reach the adapter.
        for request in ('Comprime ese programa en un ZIP llamado context.zip.',
                        'Comprime suma.py en un ZIP llamado explicit.zip.'):
            action = plan_from_text(request, c.state, requested_operations=('CREATE_ARCHIVE',))
            self.assertEqual(action.resolved_artifacts, (ref,))
            plan = c.plan_adapter.compile(action, source='FAST_PATH')
            actual = plan.typed_inputs['artifact_bindings']['input.0']
            self.assertEqual((actual.artifact_id, actual.version, actual.hash),
                             (ref.artifact_id, ref.version, ref.hash))
            c._execute(plan, turn_id='g6-integration', confirmed_effects=())
            with zipfile.ZipFile(action.arguments['destination']) as archive:
                self.assertEqual(archive.namelist(), ['suma.py'])
                self.assertEqual(archive.read('suma.py'), source.read_bytes())
            indexed = c.state.recent_artifact_state.ref_by_path(action.arguments['destination'])
            self.assertEqual(indexed.artifact_type, 'ARCHIVE')

