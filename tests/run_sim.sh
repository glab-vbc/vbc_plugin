

echo "testing valency plugin GPU"
python valency_sqd_test.py -d 'gpu' -p 'custom'

echo "testing valency plugin CPU"
python valency_sqd_test.py -d 'cpu' -p 'custom'

echo "testing hoomd sqd GPU"
python valency_sqd_test.py -d 'gpu' -p 'hoomd'

echo "testing hoomd sqd CPU"
python valency_sqd_test.py -d 'cpu' -p 'hoomd'
