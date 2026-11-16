import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch,Mock
import mask_review,seedance_bridge

class WorkflowTests(unittest.TestCase):
 def test_mask_changed_invalidates_approval(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for name in mask_review.FILES:(p/name).write_bytes(b'original')
   fingerprint=mask_review.fingerprint(p)
   mask_review.review(p,True,'Entire clip reviewed',fingerprint)
   mask_review.require_approved(p)
   (p/'mask.mp4').write_bytes(b'changed')
   with self.assertRaises(ValueError):mask_review.require_approved(p)
   with self.assertRaises(ValueError):mask_review.review(p,True,'stale',fingerprint)
 def test_existing_submission_not_requeued(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'seedance-response.json').write_text('{}')
   with patch('mask_review.require_approved'),patch('seedance_bridge.requests.post') as post:
    with self.assertRaises(ValueError):seedance_bridge.submit(p,'prompt','reference.png')
    post.assert_not_called()
 def test_failed_submission_can_be_explicitly_archived_and_retried(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'audio-manifest.json').write_text('{}')
   (p/'seedance-response.json').write_text(json.dumps({'output':{'task_id':'old-id'}}))
   (p/'provider-status.json').write_text(json.dumps({'output':{'task_status':'failed','error_code':'10001'}}))
   (p/'generation.json').write_text('{}')
   response=Mock();response.ok=True;response.json.return_value={'output':{'task_id':'new-id','task_status':'pending'}}
   with patch('mask_review.require_approved'),patch('seedance_bridge.credential',return_value='test'),patch('seedance_bridge.setting',return_value='https://example.org'),patch('seedance_bridge.upload',return_value='https://example.org/media'),patch('audio_workflow.prepare_seedance_video',return_value=p/'seedance-input.mp4'),patch('seedance_bridge._video_parameters',return_value={'resolution':'720p'}),patch('seedance_bridge.requests.post',return_value=response):
    self.assertEqual(seedance_bridge.submit(p,'Retry subject','reference.png',model='SeeDance-2-5-LumaCraft',retry_failed=True),'new-id')
   self.assertEqual(seedance_bridge.saved_task_id(p),'new-id')
   attempts=list((p/'attempts').iterdir());self.assertEqual(len(attempts),1)
   self.assertEqual(seedance_bridge.saved_task_id(attempts[0]),'old-id')
 def test_nonfailed_submission_cannot_use_retry_flag(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'audio-manifest.json').write_text('{}');(p/'seedance-response.json').write_text('{}')
   (p/'provider-status.json').write_text(json.dumps({'output':{'task_status':'running'}}))
   with patch('mask_review.require_approved'),patch('seedance_bridge.requests.post') as post:
    with self.assertRaises(ValueError):seedance_bridge.submit(p,'prompt','reference.png',model='SeeDance-2-5-LumaCraft',retry_failed=True)
    post.assert_not_called()
 def test_provider_prompt_suffix_is_not_duplicated_on_retry(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'audio-manifest.json').write_text('{}')
   expanded='User prompt'+seedance_bridge.PROMPT_SUFFIX+'The reference image supplies the replacement visual identity.'
   response=Mock();response.ok=True;response.json.return_value={'output':{'task_id':'new-id','task_status':'pending'}}
   with patch('mask_review.require_approved'),patch('seedance_bridge.credential',return_value='test'),patch('seedance_bridge.upload',return_value='https://example.org/media'),patch('audio_workflow.prepare_seedance_video',return_value=p/'seedance-input.mp4'),patch('seedance_bridge._video_parameters',return_value={'resolution':'720p'}),patch('seedance_bridge.requests.post',return_value=response) as post:
    seedance_bridge.submit(p,expanded,'reference.png',model='Seedance2.0')
   payload=post.call_args.kwargs['json'];self.assertEqual(payload['input']['prompt'].count('The reference video is the prepared source'),1)
   self.assertIn('must never be visible',payload['input']['prompt'])
   self.assertEqual(json.loads((p/'generation.json').read_text())['user_prompt'],'User prompt')
 def test_edit_draft_self_contained(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'audio-manifest.json').write_text('{}')
   response=Mock();response.ok=True;response.json.return_value={'output':{'task_id':'fake-id','task_status':'pending'}}
   with patch('mask_review.require_approved'),patch('seedance_bridge.credential',return_value='test'),patch('seedance_bridge.setting',return_value='https://example.org'),patch('seedance_bridge.upload',return_value='https://example.org/media'),patch('audio_workflow.prepare_seedance_video',return_value=p/'seedance-input.mp4'),patch('seedance_bridge._video_parameters',return_value={'resolution':'720p'}),patch('seedance_bridge.requests.post',return_value=response) as post:
    seedance_bridge.submit(p,'Replace one subject','reference.png',model='SeeDance-2-5-LumaCraft')
    payload=post.call_args.kwargs['json']
    self.assertEqual(payload['model'],'SeeDance-2-5-LumaCraft');self.assertEqual(payload['parameters']['resolution'],'720p');self.assertEqual(payload['input']['media'][0]['type'],'reference_image');self.assertEqual(payload['input']['media'][1]['type'],'reference_video')
    self.assertTrue((p/'generation.json').exists())
 def test_collection_downloads_and_restores(self):
  with tempfile.TemporaryDirectory() as d:
   response=Mock();response.json.return_value={'output':{'task_status':'succeeded','results':['https://example.org/result.mp4']}}
   media=Mock();media.content=b'video'
   with patch('seedance_bridge.credential',return_value='test'),patch('seedance_bridge.requests.get',side_effect=[response,media]),patch('seedance_bridge.finish_preview') as restore:
    seedance_bridge.wait_and_finish(d,'fake-id')
    self.assertEqual((Path(d)/'seedance-result.mp4').read_bytes(),b'video');restore.assert_called_once()
 def test_hd_only_reuses_draft(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for name in ['upload.mp4','final-preview.mp4']:(p/name).write_bytes(b'test')
   (p/'seedance-request.json').write_text(json.dumps({'model':'SeeDance-2-5-LumaCraft','input':{'prompt':'x'},'parameters':{'resolution':'720p'}}))
   response=Mock();response.ok=True;response.json.return_value={'output':{'task_id':'hd-id','task_status':'pending'}}
   with patch('seedance_bridge.credential',return_value='test'),patch('seedance_bridge.requests.post',return_value=response) as post:
    self.assertEqual(seedance_bridge.submit_hd(p),'hd-id')
    self.assertEqual(post.call_args.kwargs['json']['parameters']['resolution'],'1080p')
    with self.assertRaises(ValueError):seedance_bridge.submit_hd(p)

if __name__=='__main__':unittest.main()
