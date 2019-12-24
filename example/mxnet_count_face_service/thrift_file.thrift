namespace py count_face_service

struct TCountFaceResult {
    1: i32 errorCode,
	2: optional map<i32, i32> mapResult
}

service CountFaceService {
    TCountFaceResult count(1: map<i32, string> mapUrlUserAvt);
	void ping();
}
