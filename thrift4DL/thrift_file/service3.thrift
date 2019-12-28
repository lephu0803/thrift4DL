namespace py thrift4DL

service Thrift4DLService {
    string predict(1: string image_binary);
    void ping()
}