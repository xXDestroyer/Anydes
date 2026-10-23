"""Controlled provider faults, real structured reviewer/parser/local invariants."""
from dataclasses import replace
import json
from types import SimpleNamespace as NS
import unittest

from anydes_tools.tests.test_natural_response import review_payload
from anydes_tools.conversation.natural_response import (
    review_prose,parse_review,REVIEW_SCHEMA,VerificationStatus as S)
from anydes.hybrid.router import ProviderFailure,OnlineContextSanitizer
from anydes.realtime.transport import StreamOutcome
from anydes.realtime.validation import ValidationResult


def outcome(payload):
    text = payload if isinstance(payload,str) else json.dumps(payload)
    return StreamOutcome(text,0,0,0,0,0,0,0,0,0,'prepared')


class Client:
    model='prepared-primary'
    def __init__(self,replies):
        self.replies=iter(replies)
        self.calls=[]
    def stream(self,messages,**kwargs):
        self.calls.append((messages,kwargs))
        reply=next(self.replies)
        if isinstance(reply,Exception):
            raise reply
        return outcome(reply)


class LocalTransport(Client):
    replies=()
    instances=[]
    def __init__(self,**config):
        super().__init__(type(self).replies)
        self.config=config
        self.model=config['model']
        type(self).instances.append(self)


class ReviewerResilience(unittest.TestCase):
    def setUp(self):
        LocalTransport.instances=[]

    def router(self,primary,fallback=()):
        primary=Client(primary)
        LocalTransport.replies=fallback
        # The client's chat wrapper is deliberately forbidden. Only the raw,
        # request-local transport with the same review prompt may be called.
        local=NS(base_url='http://local.test',model='prepared-local',transport=object.__new__(LocalTransport),
                 stream=lambda *a,**k:self.fail('conversational local wrapper called'))
        router=NS(online=primary,local=local,sanitizer=OnlineContextSanitizer(),
                  config=NS(local_keep_alive='10m',local_num_ctx=768))
        return router,primary

    def run_review(self,primary,fallback=(),*,text='A present subjective response.',prior=None,
                   grounding=None):
        router,primary=self.router(primary,fallback)
        events=[]
        result=review_prose(lambda *a,**k:self.fail('conversational transport called'),
            [{'role':'system','content':'PERSONALITY PROMPT MUST NOT BE A REVIEW INSTRUCTION'}],text,{},
            record=lambda e,**k:events.append((e,k)),transport=NS(router=router),
            grounding=grounding or {'self_preferences':[]},
            validation=ValidationResult(True,text,()),prior_nonfactual=prior)
        return result,primary,events

    @staticmethod
    def unavailable():
        return ProviderFailure('rate_limit',status=429,retry_after=7)

    def test_primary_valid_structured_approval(self):
        result,primary,events=self.run_review([review_payload()])
        self.assertEqual(result.status,S.PASS)
        self.assertTrue(result.publishable)
        self.assertEqual(len(primary.calls),1)
        self.assertFalse(LocalTransport.instances)

    def test_429_fallback_same_contract_and_grounding_no_chat_wrapper(self):
        result,primary,events=self.run_review([self.unavailable()],[review_payload()])
        self.assertTrue(result.publishable)
        local=LocalTransport.instances[0]
        self.assertEqual(primary.calls[0][0],local.calls[0][0])
        self.assertGreaterEqual(local.config['num_ctx'],8192)
        self.assertEqual(len(local.calls),1)
        serialized=json.dumps(local.calls[0][0])
        self.assertNotIn('PERSONALITY PROMPT',serialized)
        schema=json.loads(local.calls[0][0][0]['content'].split('schema: ',1)[1].split('. Comprueba',1)[0])
        self.assertEqual(schema,REVIEW_SCHEMA)
        self.assertIn('self_preferences',serialized)

    def test_fallback_can_reject_after_429(self):
        result,_,_=self.run_review([self.unavailable()],
            [review_payload(['UNSUPPORTED_FACT'])])
        self.assertEqual(result.status,S.FAIL)
        self.assertFalse(result.publishable)

    def test_malformed_fallback_one_repair_then_valid(self):
        result,primary,events=self.run_review([self.unavailable()],
            ['A conversational answer, not JSON.',review_payload()])
        self.assertTrue(result.publishable)
        self.assertEqual(len(primary.calls),1)
        self.assertEqual(len(LocalTransport.instances[0].calls),2)
        self.assertEqual(sum(e=='reviewer_protocol_invalid' for e,_ in events),1)
        self.assertEqual([d['repair'] for e,d in events if e=='reviewer_provider_attempt'],
                         [False,False,True])
        self.assertNotIn('A conversational answer',str(events))

    def test_repeat_malformed_stops_after_three_total_attempts(self):
        result,primary,events=self.run_review([self.unavailable()],['bad','still bad'])
        self.assertEqual(result.status,S.INCONCLUSIVE)
        self.assertFalse(result.publishable)
        self.assertEqual(len(primary.calls)+len(LocalTransport.instances[0].calls),3)

    def test_all_unavailable_prior_nonfactual_assessment_safe_degradation(self):
        prior,_,_=self.run_review([review_payload(grounding=False)])
        result,_,events=self.run_review([self.unavailable()],
            [OSError('local unavailable')],prior=prior)
        self.assertEqual(result.status,S.INCONCLUSIVE)
        self.assertTrue(result.publishable)
        self.assertEqual(result.request_sha256,prior.request_sha256)
        self.assertTrue(any(e=='reviewer_safe_degradation' for e,_ in events))

    def test_nonfactual_but_uncertain_optional_review_can_degrade(self):
        payload={**review_payload(grounding=False),'status':'INCONCLUSIVE'}
        result,_,_=self.run_review([payload])
        self.assertTrue(result.publishable)
        self.assertFalse(result.requires_grounding)

    def test_all_unavailable_factual_claim_cannot_publish(self):
        result,_,_=self.run_review([self.unavailable()],[OSError('unavailable')],
            text='A factual assertion requiring user evidence.')
        self.assertEqual(result.status,S.INCONCLUSIVE)
        self.assertTrue(result.requires_grounding)
        self.assertFalse(result.publishable)

    def test_low_risk_certificate_cannot_cover_different_response(self):
        prior,_,_=self.run_review([review_payload(grounding=False)])
        result,_,_=self.run_review([self.unavailable()],[OSError('unavailable')],
            text='A different factual statement.',prior=prior)
        self.assertFalse(result.publishable)

    def test_critical_approval_is_not_an_unavailable_bypass(self):
        prior,_,_=self.run_review([review_payload(grounding=True)])
        result,_,_=self.run_review([self.unavailable()],[OSError('unavailable')],prior=prior)
        self.assertFalse(result.publishable)

    def test_missing_assessment_never_becomes_nonfactual_by_domain(self):
        result,_,_=self.run_review([self.unavailable()],[OSError('unavailable')])
        self.assertFalse(result.publishable)

    def test_local_protocol_invariant_precedes_provider_call(self):
        with self.assertRaisesRegex(ValueError,'local_invariant'):
            self.run_review([],text='grounding_reason: INTERNAL_METADATA')

    def test_relevant_field_strictness_and_safety_flag_consistency(self):
        for change in ({'requires_grounding':'false'},
                {'status':'APPROVE'}, {'basis':'leak'},
                {'violations':['UNSUPPORTED_FACT']}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                parse_review(json.dumps({**review_payload(),**change}),'digest')

    def test_exact_ui_original_hash_result_valid_nonfactual_reason_is_not_authority(self):
        import hashlib
        raw=json.dumps({**review_payload(grounding=False),'reason_code':'VALID'},indent=2)
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),
            'b1f419048a21a5a35df7ab46349ac2fc9fe4fff519b90904d7eed99a54a41606')
        review=parse_review(raw,'context-digest')
        self.assertTrue(review.publishable)
        self.assertFalse(review.requires_grounding)
        self.assertTrue(review.safe_to_publish_if_reviewer_unavailable)

    def test_exact_ui_repair_hash_grounding_dominates_unavailable_flag(self):
        import hashlib
        raw=json.dumps({**review_payload(),'safe_to_publish_if_reviewer_unavailable':True},
                       separators=(',',':'))
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),
            '8e52966a672bc7e64f89f2ea3452a8c3907fd59dcc0421e102eb60cba1fa3ca6')
        events=[]
        review=parse_review(raw,'context-digest',record=lambda e,**d:events.append((e,d)))
        self.assertTrue(review.publishable)
        self.assertTrue(review.requires_grounding)
        self.assertFalse(review.safe_to_publish_if_reviewer_unavailable)
        self.assertTrue(events[0][1]['authority_reduced'])
        result,_,_=self.run_review([self.unavailable()],[OSError('unavailable')],prior=review)
        self.assertFalse(result.publishable)

    def test_grounding_context_is_part_of_the_certificate(self):
        prior,_,_=self.run_review([review_payload(grounding=False)])
        result,_,_=self.run_review([self.unavailable()],[OSError('unavailable')],prior=prior,
                                  grounding={'self_preferences':[{'value':'changed'}]})
        self.assertFalse(result.publishable)

    def test_runtime_adapter_preserves_assessed_nonfactual_prose_during_outage(self):
        from anydes_tools.tests.test_natural_response import NaturalResponses
        helper=NaturalResponses()
        text='Me alegra compartir esta charla contigo.'
        candidate={'text':text,'offer':None}
        store={}
        router,_=self.router([review_payload(grounding=False)])
        first,validation,_,_=helper.run_adapter([candidate],intent='SELF_RELATIONAL',
            review_router=router,assessments=store)
        self.assertTrue(validation.valid)
        self.assertEqual(len(store),1)
        router,_=self.router([self.unavailable()],[OSError('unavailable')])
        second,validation,_,events=helper.run_adapter([candidate],intent='SELF_RELATIONAL',
            review_router=router,assessments=store)
        self.assertEqual(first.text,second.text)
        self.assertEqual(second.text,text)
        self.assertTrue(validation.valid)
        self.assertTrue(any(e=='reviewer_safe_degradation' for e,_ in events))


if __name__=='__main__':
    unittest.main()
