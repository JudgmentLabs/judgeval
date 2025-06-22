import json
import os
import sys
from typing import Any, Dict, Generator, List, Union

spec_file = sys.argv[1] if len(sys.argv) > 1 else './openapi.json'
with open(spec_file, 'r') as f:
    SPEC = json.load(f)

JUDGEVAL_PATHS: List[str] = ['/traces/save/', '/traces/fetch/']



def resolve_ref(ref: str) -> str:
    assert ref.startswith('#/components/schemas/'), 'Reference must start with #/components/schemas/'
    return ref.replace('#/components/schemas/', '')


def walk(obj: Any) -> Generator[Any, None, None]:
    if isinstance(obj, list):
        for item in obj:
            yield from walk(item)
    elif isinstance(obj, dict):
        for key in obj:
            yield obj[key]
            yield from walk(obj[key])
    else:
        yield obj


def get_referenced_schemas(obj: Any) -> Generator[str, None, None]:
    for value in walk(obj):
        if isinstance(value, dict) and '$ref' in value:
            ref = value['$ref']
            resolved = resolve_ref(ref)
            assert isinstance(ref, str), 'Reference must be a string'
            yield resolved


filtered_paths = {
    path: spec_data 
    for path, spec_data in SPEC['paths'].items() 
    if path in JUDGEVAL_PATHS
}


def filter_schemas() -> Dict[str, Any]:
    to_commit: Dict[str, Any] = {}
    
    context = {
        'paths': filtered_paths,
        'result': {},
    }
    
    while True:
        for schema in get_referenced_schemas(context):
            if schema in context['result']:
                continue
            
            assert schema in SPEC['components']['schemas'], f'Schema {schema} not found in components.schemas'
            to_commit[schema] = SPEC['components']['schemas'][schema]
        
        context['result'] = {**context['result'], **to_commit}
        if len(to_commit) == 0:
            break
        to_commit = {}
    
    return context['result']


spec = {
    'openapi': SPEC['openapi'],
    'info': SPEC['info'],
    'paths': filtered_paths,
    'components': {
        **SPEC['components'],
        'schemas': filter_schemas(),
    },
}

print(json.dumps(spec))