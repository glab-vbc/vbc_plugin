

echo "testing valency plugin GPU"
python valency_vbc_test.py -d 'gpu' -p 'custom'

echo "testing valency plugin CPU"
python valency_vbc_test.py -d 'cpu' -p 'custom'

echo "testing hoomd vbc GPU"
python valency_vbc_test.py -d 'gpu' -p 'hoomd'

echo "testing hoomd vbc CPU"
python valency_vbc_test.py -d 'cpu' -p 'hoomd'
