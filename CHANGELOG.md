# CHANGELOGS

## [2.2] - Updating

### Added

- [ - ] Documentations
- [ - ] Support logging handler
- [ x ] Support RESTFul APIs with Flask/FastAPI
- [ x ] Exception handling

### Changed

- `TModelPoolServer` has been changed to `server`
- `Thrift4DLBase` has been changed to `thriftbase`

## [2.1] - Released

### Added

- [ x ] All handlers have its own queue
- [ x ] Exception handling (Not Fully Implemented)
- [ x ] Add `ping` function

### Changed

- `Thrift4DLService` has been renamed to `connectors`

### Fixed 

- [ x ] Batches request
- [ x ] Reduce CPU Utilization while idling

## Removed

- [ x ] Some Deep Learning examples

## [2.0] - Finalized

### Added

- [ x ] Deliver and Receiver works in a pipe
- [ x ] Batches Request
- [ x ] Some examples in real production


## [1.1] - Released - 23 - 12 - 2019

### Added

- [ x ] Support json string protocol

## [1.0] - Released - 23 - 12 - 2019

### Added

- [ x ] Deliver and Receiver works separately
- [ x ] Shared Queue for all handlers
- [ x ] Deliver and Receiver classes

