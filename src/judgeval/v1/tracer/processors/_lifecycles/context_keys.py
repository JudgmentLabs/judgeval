from __future__ import annotations

from opentelemetry.context import create_key
from judgeval.judgment_attribute_keys import AttributeKeys


CUSTOMER_ID_KEY = create_key(AttributeKeys.JUDGMENT_CUSTOMER_ID)
SESSION_ID_KEY = create_key(AttributeKeys.JUDGMENT_SESSION_ID)
AGENT_ID_KEY = create_key(AttributeKeys.JUDGMENT_AGENT_ID)
PARENT_AGENT_ID_KEY = create_key(AttributeKeys.JUDGMENT_PARENT_AGENT_ID)
AGENT_CLASS_NAME_KEY = create_key(AttributeKeys.JUDGMENT_AGENT_CLASS_NAME)
AGENT_INSTANCE_NAME_KEY = create_key(AttributeKeys.JUDGMENT_AGENT_INSTANCE_NAME)
PROJECT_ID_OVERRIDE_KEY = create_key(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE)
