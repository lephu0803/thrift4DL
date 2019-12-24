namespace py thrift4DL

struct TResult {
    1: i32 error_code = -1,
    2: optional string error_message,
    3: optional string response, # Json
}

service Thrift4DLService {
    TResult predict(1: string request);
}