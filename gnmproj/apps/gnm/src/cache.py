import collections
import hashlib
import json

from django.core.cache.backends.db import DatabaseCache


class HashedDatabaseCache(DatabaseCache):

    def make_key(self, key, version=None):

        if not isinstance(key, collections.Hashable):
            key = json.dumps(key).encode('utf-8')
        else:
            key = str(key).encode('utf-8')

        key = hashlib.md5(key).digest()

        return super(HashedDatabaseCache, self).make_key(key, version=version)
