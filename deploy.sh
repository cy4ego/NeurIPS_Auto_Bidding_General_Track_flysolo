usage () {
    echo 
    echo "Command examples"
    echo "'sh ./deploy.sh -t tagname'"
    echo "tagname:"
    echo "  1) b4_iql_base for implicit q-learning"
    echo "  2) b4_bc_base for behavior cloning"
    echo "  3) b4_onlineLP_base for online linear programming"
    echo 
}

if [ $# -eq 0 ]
    then
        echo "No arguments supplied"
        usage 
        exit 1
fi 

while getopts t: flag
do
    case "${flag}" in 
        t) tagname=${OPTARG};;
        *) usage ; exit 1 ;;
    esac
done

echo "tagname=${tagname}"

docker build -t registry-intl.cn-beijing.aliyuncs.com/nips2024-gt-b4/nips2024env:${tagname} -f ./Dockerfile .
docker push registry-intl.cn-beijing.aliyuncs.com/nips2024-gt-b4/nips2024env:${tagname}
