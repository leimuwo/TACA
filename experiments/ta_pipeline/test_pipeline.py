import importlib.util
import pathlib
import unittest

PATH = pathlib.Path(__file__).resolve().parent / 'pipeline.py'
spec = importlib.util.spec_from_file_location('ta_local_pipeline', PATH)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PipelineTest(unittest.TestCase):
    def test_formats_and_preserve_command(self):
        command = 'echo "pytest" > script.sh'
        r = p.normalize({'unit_id': 'x', 'thought': 'Run tests', 'window_complete': True,
                         'actions': [{'id': 'a', 'function': {'name': 'bash', 'arguments': __import__('json').dumps({'command': command})}}]})
        self.assertEqual(r['actions'][0]['arguments']['command'], command)
        self.assertEqual(r['actions'][0]['action_id'], 'a')

    def test_no_silent_incomplete_window(self):
        i = [{'intent_id': 'I1', 'temporal_scope': 'current'}]
        a = [{'action_id': 'A1'}]
        pair = [{'intent_id': 'I1', 'action_id': 'A1', 'operation_match': 'no'}]
        self.assertEqual(p.aggregate(i, a, pair, False)['step_label'], 'uncertain')
        self.assertEqual(p.aggregate(i, a, pair, True)['step_label'], 'fully_inconsistent')
        self.assertEqual(p.aggregate(i, [], [], True)['step_label'], 'uncertain')

    def test_many_to_many_and_future(self):
        i = [{'intent_id': 'I1', 'temporal_scope': 'current'}, {'intent_id': 'I2', 'temporal_scope': 'current'}, {'intent_id': 'I3', 'temporal_scope': 'future'}]
        pair = [{'intent_id': k, 'action_id': 'A1', 'operation_match': 'yes'} for k in ['I1', 'I2']]
        r = p.aggregate(i, [{'action_id': 'A1'}], pair, True)
        self.assertEqual(r['step_label'], 'fully_consistent')
        self.assertEqual(len(r['intent_results']), 2)
        self.assertEqual(p.aggregate(i[2:], [], [], True)['step_label'], 'not_applicable')

    def test_partial_and_uncertainty(self):
        i = [{'intent_id': k, 'temporal_scope': 'current'} for k in ['I1', 'I2', 'I3']]
        pair = [{'intent_id': k, 'action_id': 'A1', 'operation_match': state} for k, state in [('I1', 'yes'), ('I2', 'no'), ('I3', 'uncertain')]]
        self.assertEqual(p.aggregate(i, [{'action_id': 'A1'}], pair, True)['step_label'], 'partially_consistent')
        self.assertEqual(p.aggregate(i[:1], [{'action_id': 'A1'}], [], True)['step_label'], 'uncertain')

    def test_extraction_failure_is_not_empty_intents(self):
        class Failed:
            def extract(self, thought):
                return None, {'raw': '{}', 'status': 'invalid_schema'}
        row = p.normalize({'unit_id': 'x', 'thought': 'Read file', 'actions': [], 'window_complete': True})
        result = p.run_unit(row, Failed())
        self.assertEqual(result['step_label'], 'uncertain')
        self.assertEqual(result['status'], 'technical_failure')
        self.assertIn('extraction', result)

    def test_end_to_end_mock(self):
        class Fake:
            def extract(self, thought):
                return [{'intent_id': 'I1', 'intent_quote': thought, 'temporal_scope': 'current'}], {'status': 'valid', 'raw': thought}
            def match(self, intents, actions):
                return [{'intent_id': 'I1', 'action_id': 'A1', 'operation_match': 'yes'}]
        row = p.normalize({'unit_id': 'x', 'thought': 'Read file', 'actions': [{'tool': 'read', 'args': {'path': 'f'}}], 'window_complete': True})
        result = p.run_unit(row, Fake())
        self.assertEqual(result['step_label'], 'fully_consistent')
        self.assertFalse(result['semantic_acceptance_passed'])

    def test_duplicate_action_ids_rejected(self):
        with self.assertRaises(ValueError):
            p.normalize({'unit_id': 'x', 'thought': '', 'window_complete': True, 'actions': [{'id': 'a', 'tool': 'read', 'args': {}}] * 2})


if __name__ == '__main__':
    unittest.main()
