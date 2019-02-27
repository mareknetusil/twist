PYTHON=python

echo '======RUNNING TESTS======'

echo '------RUNNING pull.py------'
for i in {0..8}; do $PYTHON pull.py 200.0 $i; done > tests_pull.txt

echo '------RUNNING hetero_pull.py------'
for i in {0..8}; do $PYTHON hetero_pull.py 200.0 $i; done > tests_hetero_pull.txt

echo '------RUNNING twist.py------'
for i in {0..8}; do $PYTHON twist.py $i; done > tests_twist.txt

echo '======TESTS DONE======'
