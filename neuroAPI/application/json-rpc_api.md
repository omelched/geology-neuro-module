# JSON-RPC API

`geology-neuro-module` json-rpc api.
This api follows the json-rpc 2.0 specification. More information available at http://www.jsonrpc.org/specification.

**Version 1.0**

## Content

- [service.echo](#service.echo)

## Endpoints

### service.echo

Echoes you back.

#### Description

--

#### Parameters

| Name            | Type   | Constraints   | Description                                   |
| --------------- | ------ | ------------- | --------------------------------------------- |
| params          | array  |               |                                               |
| params[]        | string |               | Any string to echo back                       |

#### Result

| Name                 | Type   | Constraints    | Description                               |
| -------------------- | ------ | -------------- | ----------------------------------------- |
| result               | array  |                |                                           |
| result[]             | string |                | Echoed string                             |

#### Errors

| Code | Message            | Description                           |
| ---- | ------------------ | ------------------------------------- |
|      |                    |                                       |

#### Examples

##### Request

```json
{
  "jsonrpc": "2.0",
  "id": "1234567890",
  "method": "service.echo",
  "params": ["test", "test2"]
}
```

##### Response

```json
{
  "jsonrpc": "2.0",
  "id": "1234567890",
  "result": ["test", "test2"]
}
```
