namespace py simple_vision_example

struct TVisionResult {
    1: i32 error_code = -1,
    2: optional string error_message,
    3: optional string response, # Json
}

service SimpleVisionService {
    TVisionResult predict(1: string image_binary);
    void ping();
}
