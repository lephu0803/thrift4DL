all:
	cd cython/; python setup.py build_ext --inplace; rm -rf build; cd ../
	cd pycocotools/; python setup.py build_ext --inplace; rm -rf build; cd ../
clean:
	cd cython/; rm *.so *.c *.cpp; cd ../
	cd pycocotools/; rm *.so; cd ../
