pipeline {
    agent { label 'built-in' }

    environment {
        REPO_URL = 'https://github.com/RyanF139/Face_Recognition.git'
        BRANCH = 'main'
        ENV_SOURCE = '/opt/config/face-recognition/.env'

        APP_NAME = 'face-recognition'
        IMAGE_NAME = 'face-recognition-app'
        IMAGE_TAG = 'latest'
        FULL_IMAGE = "${IMAGE_NAME}:${IMAGE_TAG}"

        DATA_ROOT = '/opt/data/face-recognition'
        FACE_LIB_PATH = '/opt/data/face-recognition/face_library'
        DB_PATH = '/opt/data/face-recognition/db.json'
        BACKUP_PATH = '/opt/data/face-recognition/backup'
    }

    stages {

        stage('Checkout Source') {
            steps {
                git branch: "${BRANCH}",
                    url: "${REPO_URL}",
                    credentialsId: '001'
            }
        }

        stage('Prepare Environment File') {
            steps {
                sh '''
                echo "Copying .env file..."
                cp $ENV_SOURCE .env
                '''
            }
        }

        stage('Prepare Persistent Storage') {
            steps {
                sh '''
                echo "Preparing persistent storage..."

                mkdir -p $FACE_LIB_PATH
                mkdir -p $BACKUP_PATH

                if [ ! -f $DB_PATH ]; then
                    echo "{}" > $DB_PATH
                fi
                '''
            }
        }

        stage('Backup Database') {
            steps {
                sh '''
                echo "Creating database backup..."

                if [ -f $DB_PATH ]; then
                    cp $DB_PATH $BACKUP_PATH/db_$(date +%F_%H-%M-%S).json
                fi
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                echo "Building Docker image..."
                docker build -t $FULL_IMAGE .
                '''
            }
        }

        stage('Stop Old Container') {
            steps {
                sh '''
                echo "Stopping old container if exists..."
                docker stop $APP_NAME 2>/dev/null || true
                docker rm $APP_NAME 2>/dev/null || true
                '''
            }
        }

        // 🔥 TAMBAHAN CLEANUP DOCKER > 48 JAM
        stage('Docker Cleanup (older than 2 days)') {
            steps {
                sh '''
                echo "Cleaning Docker resources older than 48h..."

                # container lama
                docker container prune -f --filter "until=48h" || true

                # image tidak terpakai
                docker image prune -a -f --filter "until=48h" || true

                # build cache
                docker builder prune -a -f --filter "until=48h" || true
                '''
            }
        }

        stage('Run New Container') {
            steps {
                sh '''
                echo "Running new container..."

                docker run -d \
                  --name $APP_NAME \
                  --env-file .env \
                  -v $FACE_LIB_PATH:/app/face_library \
                  -v $DB_PATH:/app/db.json \
                  -p 8000:8000 \
                  --restart=always \
                  -e TZ=Asia/Jakarta \
                  $FULL_IMAGE
                '''
            }
        }

        stage('Health Check') {
            steps {
                sh '''
                echo "Waiting container to start..."
                sleep 5

                docker ps | grep $APP_NAME
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Deploy sukses - Data tetap aman 🚀'
        }
        failure {
            echo '❌ Deploy gagal - cek log Jenkins'
        }
        always {
            echo 'Pipeline completed.'
        }
    }
}