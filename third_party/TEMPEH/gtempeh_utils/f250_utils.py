from google3.vr.perception.holobooth.data_management.python import holobooth_take_contents
from google3.vr.perception.ubercapture.data_management.python import f250_data_context
from google3.vr.perception.ubercapture.data_management.python import f250_service_type_helper

NAMESPACE_ID = 'YXJkYXRnZW4'
PARTITION_ID = 'aW5jcGlpX3A'


def get_session_id_from_scene_id(scene_id: str):
  data_context = f250_data_context.F250DataContext.CreateFromInstanceType()
  scene_locator = f250_service_type_helper.AsEncodedF250Locator(
      f250_service_type_helper.MakeF250ResourceLocator(
          NAMESPACE_ID, PARTITION_ID, scene_id
      )
  )
  scene_accessor = holobooth_take_contents.HoloboothSceneTakeContentsAccessor(
      data_context, scene_locator
  )
  sequence_accessor = scene_accessor.GetSequenceContentsAccessor()
  session_accessor = sequence_accessor.GetSessionContentsAccessor()

  session_locator = session_accessor.GetResourceLocator()
  session_id = session_locator.resource_id

  return session_id
