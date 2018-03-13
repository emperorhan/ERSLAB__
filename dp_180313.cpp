#include <cstdio>
#include <random>
#include <ctime>
#include <vector>
#include <iostream>
#include "tsa.h"
#include <string>
#include <time.h>

using namespace std;

#define VIDEO                   1000        // 비디오 수
#define END_SIMULATION          10000       // 시뮬레이션 시간, 24h
#define ARRIVAL_RATE            1.0         // 1초당 리퀘스트 수
#define THETA                   0.271       // zipf theta
#define LENGTH                  2           // L
#define SEGMENT_PARTIAL_SIZE    30          // segment zipf 분포를 만들때 몇개씩 묶어서 생성할 것 인가

#define SETTING_DISK            20          // 12
#define WORKLOAD                0
#define POPOULARITY_UPSCALE     1

int workload = WORKLOAD;

int NUMBER_OF_DISK;
double S_limit;

int                                 totalVideoSize;
int                                 videoSize[VIDEO];               // 비디오별 파일 크기
// int                                 videoLength[VIDEO];             // ex) 7200s
vector<double>                      videoPopularity;                // 비디오별 확률
int                                 videoSegmentSize[VIDEO];        // 비디오별 세그먼트 개수
vector<double>                      popularitySegment[VIDEO];       // 비디오별 세그먼트 확률
vector<int>                         videoSegmentIdx[VIDEO];         // 비디오별 세그먼트 인덱스(ssim 관련)
vector<vector<int> >                hotStoredVersion[VIDEO];        // Hot disk에 저장할 세그먼트별 버전
vector<vector<pair<int, int> > >    segment_allocated_to_disk;      // 각 디스크 마다 어떤 비디오, 세그먼트를 저장할 지
vector<int>                         segment_allocation_disk[VIDEO]; // 비디오별 세그먼트가 할당 될 disk number
vector<int>                         poisson;                        // 포아송

vector<double>                      remainDiskCapacity;             // 디스크별 남은 용량

double                              totalCapacity;                  // 사용될 총 용량

List                                *requestList;                   // request queue

double                              ErsQoE;                         // ERS QoE
double                              OriginalQoE;                    // ALL QoE
int                                 serviceCount;                   // cnt

int                                 maximumCapacity;                // 모든 버전을 저장할
int                                 maximumDiskNumber;              // 모든 버전을 저장 했을 때의 디스크 개수

int                                 changeCount;

double                              ErsPower;
double                              OriginalPower;

double                              ErsRequestServiceTime;
double                              OriginalRequestServiceTime;

vector<double>                      diskSegmentPopularitySum;

double                              al1Upperbound;
double                              al1Profit;

double***                           dp;
bool***                             visited;
pair<int, int>***                   parent;


vector<double>                      algo3QoE;
vector<vector<pair<int, pair<int, pair<int, int> > > > >    diskRequestSegment;

default_random_engine engine(static_cast<unsigned int>(time(0)));

// uniform_int_distribution<unsigned int> Video_Length_GEN(0, 3600);
uniform_int_distribution<unsigned int> Request_Version_Randnum(1, 100);
uniform_int_distribution<unsigned int> Rand(1, 1000000 * POPOULARITY_UPSCALE);
// uniform_int_distribution<unsigned int> RandSize(120, 130);


int VideoNum(int video){
    if(video < (VIDEO / 5) * 1){
        return 102;
    }
    else if(video < (VIDEO / 5) * 2){
        return 224;
    }
    else if(video < (VIDEO / 5) * 3){
        return 512;
    }
    else{
        return 666;
    }
}

void InitSsim() {
    for (int i = 0; i < VERSION_SIZE; i++) {
        for (int j = 0; j < 667; j++) {
            double temp = ssim[i][j];
            if (temp >= 0.99) ssim[i][j] = 5.0;
            else if (temp >= 0.95) ssim[i][j] = 25.0 * temp - 19.75;
            else if (temp >= 0.88) ssim[i][j] = 14.29 * temp - 9.57;
            else if (temp >= 0.5) ssim[i][j] = 3.03 * temp + 0.48;
            else ssim[i][j] = 1;
        }
    }
}

void InitPopularity(){
    for(int video = 0; video < VIDEO; video++){
        double partialSum = 0;
        double segPopularitySum = 0;
        for(int seg = 0; seg < videoSegmentSize[video]; seg += SEGMENT_PARTIAL_SIZE){
            double currentPopularity = popularitySegment[video][seg];
            for(int idx = 0; idx < SEGMENT_PARTIAL_SIZE; idx++){
                popularitySegment[video][seg + idx] = 1 - partialSum;
            }
            segPopularitySum += popularitySegment[video][seg];
            partialSum += currentPopularity;
        }
        for(int seg = 0; seg < videoSegmentSize[video]; seg += SEGMENT_PARTIAL_SIZE){
            for(int idx = 0; idx < SEGMENT_PARTIAL_SIZE; idx++){
                popularitySegment[video][seg + idx] = (popularitySegment[video][seg + idx] * POPOULARITY_UPSCALE) / segPopularitySum;
            }
        }
    }
}

void PrintPopularity(){
    for(int video = 0; video < VIDEO; video++){
        printf("video: %d\n", video);
        double popSum = 0;
        for(int seg = 0; seg < videoSegmentSize[video]; seg += SEGMENT_PARTIAL_SIZE){
//            printf("segment: %d ---> popularity: %lf\n", seg, popularitySegment[video][seg]);
            printf("segment: %d ---> popularity: %lf\n", seg, videoPopularity[video] * popularitySegment[video][seg]);
            popSum += popularitySegment[video][seg];
        }
        printf("sum of segment popularity %lf\n", popSum);
    }
}

void InitVideoSize(){
    for(int video = 0; video < VIDEO; video++){
        // int videoSizePerMinute = RandSize(engine);
        int videoSizePerMinute = 125;
        videoSize[video] = videoSizePerMinute * (videoLength[video]/60);
        totalVideoSize += videoSize[video];
    }
    // printf("%d\n", totalVideoSize);
}

void GetMaximumDiskNumber(){
    for(int ver = 0; ver < VERSION_SIZE; ver++) maximumCapacity += totalVideoSize * versionCapacityRatio[ver];
    maximumDiskNumber = maximumCapacity / (DISK * 0.99);
    if(maximumCapacity % (int)(DISK * 0.99)) maximumDiskNumber++;
}

double GetChunkCapacity(int video, int seg, int ver){
    return ((double)videoSize[video] / (double)videoSegmentSize[video]) * versionCapacityRatio[ver];
}

double GetChunkPopularity(int video, int seg, int ver){
    return videoPopularity[video] * popularitySegment[video][seg] * workloads[workload][ver];
}

double GetQoEGain(int video, int seg, int ver){
    return GetChunkPopularity(video, seg, ver) * ssim[ver][videoSegmentIdx[video][seg]];
}

double GetServiceTime(int video, int seg, int ver){
    return (((double)GetChunkCapacity(video, seg, ver) / TRANSFER_TIME) + SEEK_TIME);
}

void printCapacity(double cap){
    int     TB = 0;
    int     GB = 0;
    double  MB = cap;

    GB = MB / 1000.0;
    MB -= GB*1000;
    TB = GB / 1000.0;
    GB %= 1000;

    if(TB) printf("%3.dTB  ", TB);
    if(GB) printf("%3.dGB  ", GB);
    printf("%3.3lfMB\n", MB);
}

void ZipfDistribution(vector<double> &vec, int size, double theta){
    double gFactor = 0;

    for(int i = 1; i <= size; i++) gFactor += 1 / pow(i, theta);

    gFactor = 1.0 / gFactor;

    for(int i = 0; i < size; i++) vec.push_back(gFactor / pow(i + 1, theta));
}

void ZipfDistributionPartialSum(vector<double> &vec, int vec_size, int partial_size, double theta){
    int zipf_size = vec_size / partial_size;
    if(vec_size%partial_size) zipf_size++;

    double gFactor = 0;

    for(int i = 1; i <= zipf_size; i++) gFactor += 1 / pow(i, theta);

    gFactor = 1.0 / gFactor;


    for(int i = 0; i < zipf_size; i++){
        double tmp = gFactor / pow(i + 1, theta);
        int ssize = partial_size;
        if((i + 1)*partial_size > vec_size) ssize -= ((i + 1)*partial_size)%vec_size;
        for(int cnt = 0; cnt < ssize; cnt++){
            vec.push_back(tmp);
        }
    }
}

void SetNumberOfDisk(int numberOfDisk){
    NUMBER_OF_DISK = numberOfDisk;
    S_limit = (DISK * NUMBER_OF_DISK) * 0.99;
    segment_allocated_to_disk.assign(NUMBER_OF_DISK, vector<pair<int, int> >());
    remainDiskCapacity.assign(NUMBER_OF_DISK, 0);
    printf("Number of disk : %d, Storage limit : ", NUMBER_OF_DISK);
    printCapacity((double)S_limit);
}

void minimumDiskServiceTime(){
    double T_min = 0;

    double videoAverageTime[VIDEO] = {};

    for(int video = 0; video < VIDEO; video++){
        for(int seg = 0, idx = 1; seg < videoSegmentSize[video]; seg += SEGMENT_PARTIAL_SIZE){
            videoAverageTime[video] += videoPopularity[video] * popularitySegment[video][seg] * idx * LENGTH * SEGMENT_PARTIAL_SIZE;
            idx++;
        }
    }
    for(int video = 0; video < VIDEO; video++){
        T_min += (ARRIVAL_RATE * videoAverageTime[video]) * GetServiceTime(video, 0, VERSION_SIZE-1);
    }
    printf("T_min : %5.3lf, L : %5.d\n", T_min, LENGTH);
    int numberOfDisk = T_min / LENGTH;
    if((T_min - (numberOfDisk * LENGTH)) > 0) numberOfDisk++;
    SetNumberOfDisk(numberOfDisk);
}

void Heuristic_find_X(){
    vector<pair<double, pair<int, pair<int, int> > > > vec;
    for(int v = 0; v < VIDEO; v++) hotStoredVersion[v].assign(videoSegmentSize[v], vector<int>());
    double current_capacity = 0;
    for(int video = 0; video < VIDEO; video++){
        for (int seg = 0; seg < videoSegmentSize[video]; seg++) {
            for (int ver = 0; ver < VERSION_SIZE; ver++) {
                if(ver==0 || ver==VERSION_SIZE-1){
                    current_capacity += GetChunkCapacity(video, seg, ver);
                    hotStoredVersion[video][seg].push_back(ver);
                }
                else vec.push_back({-(GetQoEGain(video, seg, ver) / GetChunkCapacity(video, seg, ver)), {video, {seg, ver}}});
            }
        }
    }
    sort(vec.begin(), vec.end());
    printf("base chunk video cap is ");
    printCapacity(current_capacity);
    if(current_capacity > S_limit){
        printf("Not enough Storage Limit!\n");
        int numberOfDisk = NUMBER_OF_DISK;
        while(current_capacity > (numberOfDisk * DISK) * 0.99) numberOfDisk++;
        SetNumberOfDisk(numberOfDisk);
    }
    int vec_idx = 0;
    while(vec_idx != vec.size()){
        double  profit  = -vec[vec_idx].first;
        int     video   = vec[vec_idx].second.first;
        int     seg     = vec[vec_idx].second.second.first;
        int     ver     = vec[vec_idx].second.second.second;
        if(S_limit < current_capacity + GetChunkCapacity(video, seg, ver)) {
            al1Upperbound += ((S_limit - current_capacity) / (GetChunkCapacity(video, seg, ver))) * profit;
            break;
        }
        al1Upperbound    += profit;
        al1Profit        += profit;
        current_capacity += GetChunkCapacity(video, seg, ver);
        hotStoredVersion[video][seg].push_back(ver);
        vec_idx++;
    }
    for(int video = 0; video < VIDEO; video++)
        for(int seg = 0; seg < videoSegmentSize[video]; seg++)
            sort(hotStoredVersion[video][seg].begin(), hotStoredVersion[video][seg].end());
    printf("Current Capacity : ");
    printCapacity((double)current_capacity);
}

void chunk_allocation(){
    diskSegmentPopularitySum.assign(NUMBER_OF_DISK, 0);

    vector<pair<double, pair<double, pair<int, int> > > > valuable_segment;
    for(int video = 0; video < VIDEO; video++){
        segment_allocation_disk[video].assign(videoSegmentSize[video], 0);      // 비디오별 세그먼트가 어디 디스크에 저장 되었는지
        for(int seg = 0; seg < videoSegmentSize[video]; seg++){
            double hot_segment_popularity   = 0;
            double hot_segment_capacity     = 0;
            for(auto ver : hotStoredVersion[video][seg]){
                hot_segment_popularity  += GetChunkPopularity(video, seg, ver);
                hot_segment_capacity    += GetChunkCapacity(video, seg, ver);
            }
            valuable_segment.push_back({hot_segment_popularity / hot_segment_capacity, {hot_segment_capacity, {video, seg}}});
        }
    }
    sort(valuable_segment.begin(), valuable_segment.end());
    vector<pair<double, pair<double, int> > > disk;                             // value, capacity, disk_number
    for(int d = 0; d < NUMBER_OF_DISK; d++){
        disk.push_back({0, {DISK, d}});                                         // 디스크 초기화
        remainDiskCapacity[d] = DISK;
    }
    while(valuable_segment.size()){                                             // 세그먼트의 가치를 담은 벡터 data가 있으면
        double value    = -valuable_segment.back().first;
        double capacity = valuable_segment.back().second.first;
        int video       = valuable_segment.back().second.second.first;
        int seg         = valuable_segment.back().second.second.second;
        valuable_segment.pop_back();

        while(disk.size()){                                                     // 빈 디스크가 있으면
            if(disk.back().second.first - capacity >= 0) break;                 // 현재 디스크에 현재 세그먼트 버전 셋이 저장될 수 없으면
            else disk.pop_back();                                               // 디스크를 제거한다
        }
        if(disk.empty()) break;

        int disk_number = disk.back().second.second;
        disk.back().first               += value;                               // 디스크 밸류에 +
        disk.back().second.first        -= capacity;                            // 현재 디스크의 남은 용량에서 현재 세그먼트 버전 셋을 뺀다
        remainDiskCapacity[disk_number] -= capacity;
        totalCapacity                   += capacity;
        sort(disk.begin(), disk.end());                                         // 디스크 밸류가 최하인 디스크에 할당하기 위해 소팅

        segment_allocation_disk[video][seg] = disk_number;                      // 비디오별 세그먼트가 할당 될 디스크 넘버 저장
        // 여기서 세그멘테이션 오류 발생했었음 - data가 많아 지면! --> 해결 위에 디스크 코딩 잘못함.
        segment_allocated_to_disk[disk_number].push_back({video, seg});         // 할당 받은 디스크 벡터에 비디오, 세그먼트 넘버 저장
        diskSegmentPopularitySum[disk_number] += videoPopularity[video] * popularitySegment[video][seg];
    }
}

void print_disk_seg(){
    for(int d = 0; d < NUMBER_OF_DISK; d++){
        printf("----------------------------------------------------------\nDISK NUM : %d size : %lu, remain disk cap : %lf\n", d, segment_allocated_to_disk[d].size(), remainDiskCapacity[d]);
        for(auto pair_v_s : segment_allocated_to_disk[d]){
            int video   = pair_v_s.first;
            int segment = pair_v_s.second;
            printf("video : %5.1d -- seg : %5.1d -- ", video, segment);
            for(auto ver : hotStoredVersion[video][segment]) printf("%d, ", ver);
            printf("\n");
        }
        printf("\n");
    }
}

void printX(){
    for(int video = 0; video < VIDEO; video++){
        printf("video : %3.1d\n----------------------------------------------------------\n", video);
        for(int seg = 0; seg < videoSegmentSize[video]; seg++){
            printf("| seg : %5.1d | ver => ", seg);
            for(auto ver : hotStoredVersion[video][seg]){
                printf("%d, ", ver);
            }
            printf("\n");
        }
    }
}

int GetRequestSegment(Request *req, int sec){
    int ret     = (int)((sec - req->startTime) / LENGTH);
    int size    = videoSegmentSize[req->video];
    if(ret >= size) ret = size - 1;
    return ret;
}

string GetKey(int video, int seg){
    return to_string(video) + " " + to_string(seg);
}

double GetDPvalue(int diskNum, int bth, int reqSeg, int curVer){
    if(bth > LENGTH * 10000) return INF;
    if(reqSeg == diskRequestSegment[diskNum].size()) return 0;
    double& ret = dp[bth][reqSeg][curVer];
    bool& isVisited = visited[bth][reqSeg][curVer];
    if(isVisited) return ret;
    isVisited = true;
    int video = diskRequestSegment[diskNum][reqSeg].first;
    int seg = diskRequestSegment[diskNum][reqSeg].second.first;
    int reqVer = diskRequestSegment[diskNum][reqSeg].second.second.first;
    ret = ssim[curVer][videoSegmentIdx[video][seg]];
    double maxValue = 0;
    for(auto ver : hotStoredVersion[video][seg]){
        if(reqVer <= ver){
            int nextBandwidth = bth + (int)(GetServiceTime(video, seg, ver) * 10000);
            double value = GetDPvalue(diskNum, nextBandwidth, reqSeg + 1, ver);
            if(maxValue < value){
                maxValue = value;
                parent[bth][reqSeg][curVer].first = ver;
                parent[bth][reqSeg][curVer].second = nextBandwidth;
            }
        }
    }
    return ret += maxValue;
}

void bandwidth_allocation(int sec, int client){
    printf("%d client do it!\n", client);
    vector<pair<int, pair<int, pair<int, pair<int, bool> > > > > service; // video, seg, ver, selected_version_index, seek
    diskRequestSegment.assign(NUMBER_OF_DISK, vector<pair<int, pair<int, pair<int, int> > > >());

    Request *pos    = requestList->GetHead()->next;
    Request *tail   = requestList->GetTail();
    int serviceNum  = 0;
    while(pos != tail){
        int video               = pos->video;
        int seg                 = GetRequestSegment(pos, sec);
        int ver                 = pos->ver;
        diskRequestSegment[segment_allocation_disk[video][seg]].push_back({video, {seg, {ver, serviceNum}}});
        service.push_back({video, {seg, {ver, {-1, pos->seek}}}});
        pos->seek = !pos->seek;
        pos = pos->next;
        serviceNum++;
    }
    dp = new double**[20001];
    visited = new bool**[20001];
    parent = new pair<int, int>**[20001];
    for(int d = 0; d < NUMBER_OF_DISK; d++){
        for(int leng = 0; leng <= 20000; leng++) {
            dp[leng] = new double *[diskRequestSegment[d].size()];
            visited[leng] = new bool *[diskRequestSegment[d].size()];
            parent[leng] = new pair<int, int> *[diskRequestSegment[d].size()];
            for (int reqSeg = 0; reqSeg < diskRequestSegment[d].size(); reqSeg++) {
                dp[leng][reqSeg] = new double[VERSION_SIZE];
                visited[leng][reqSeg] = new bool[VERSION_SIZE];
                parent[leng][reqSeg] = new pair<int, int>[VERSION_SIZE];
                for(int v = 0; v < VERSION_SIZE; v++){
                    visited[leng][reqSeg][v] = 0;
                    parent[leng][reqSeg][v].first = -1;
                }
            }
        }

        double maxVal = 0;
        int curVer;
        int curBandwidth;
        int video = diskRequestSegment[d][0].first;
        int seg = diskRequestSegment[d][0].second.first;
        int reqVer = diskRequestSegment[d][0].second.second.first;
        for(auto v : hotStoredVersion[video][seg]){
            if(reqVer <= v){
                int bth = (int)(GetServiceTime(video, seg, v) * 10000);
                double dpVal = GetDPvalue(d, bth, 0, v);
                if(dpVal < 0) printf("???\n");
                if(maxVal < dpVal){
                    maxVal = dpVal;
                    curVer = v;
                    curBandwidth = bth;
                }
            }
        }
        algo3QoE[d] += maxVal;

        serviceNum = diskRequestSegment[d][0].second.second.second;
        service[serviceNum].second.second.second.first = curVer;

        int prevVer, prevBandwidth;
        // for(int reqSeg = 1; reqSeg < diskRequestSegment[d].size(); reqSeg++){
        //     prevVer = curVer;
        //     prevBandwidth = curBandwidth;
        //     curVer = parent[prevBandwidth][reqSeg][prevVer].first;
        //     curBandwidth = parent[prevBandwidth][reqSeg][prevVer].second;
        //     serviceNum = diskRequestSegment[d][reqSeg].second.second.second;
        //     service[serviceNum].second.second.second.first = curVer;
        //     if(curVer == -1){
        //         printf("disk %d bandwidth err --- reqSeg: %d\n", d, reqSeg);
        //         exit(0);
        //     }
        // }
        for(int reqSeg = 0; reqSeg < diskRequestSegment[d].size(); reqSeg++){
            prevVer = curVer;
            prevBandwidth = curBandwidth;
            serviceNum = diskRequestSegment[d][reqSeg].second.second.second;
            service[serviceNum].second.second.second.first = curVer;
            if(curVer == -1){
                printf("disk %d bandwidth err --- reqSeg: %d\n", d, reqSeg);
                exit(0);
            }
            curVer = parent[prevBandwidth][reqSeg][prevVer].first;
            curBandwidth = parent[prevBandwidth][reqSeg][prevVer].second;
        }
        for(int leng = 0; leng <= 20000; leng++) {
            for (int reqSeg = 0; reqSeg < diskRequestSegment[d].size(); reqSeg++) {
                delete[] dp[leng][reqSeg];
                delete[] visited[leng][reqSeg];
                delete[] parent[leng][reqSeg];
            }
            delete[] dp[leng];
            delete[] visited[leng];
            delete[] parent[leng];
        }
//        delete[] dp;
//        delete[] visited;
//        delete[] parent;
    }


    ErsRequestServiceTime       = 0;
    OriginalRequestServiceTime  = 0;
    // printf("client: %d service start\n", client);
    int servCount = 1;
    for(auto sv : service){
        int video               = sv.first;
        int seg                 = sv.second.first;
        int ver                 = sv.second.second.first;
        int seek                = sv.second.second.second.second;
        int selectedVersion     = sv.second.second.second.first;

        // printf("request version %d selected version %d\n", ver, selectedVersion);

        ErsQoE      += ssim[selectedVersion][videoSegmentIdx[video][seg]];
        OriginalQoE += ssim[ver][videoSegmentIdx[video][seg]];

        // printf("%d serv QoE %lf\n", servCount, ssim[selectedVersion][videoSegmentIdx[video][seg]]);

        if(seek){
            ErsPower        += (SEEK_TIME * SEEK_POWER);
            OriginalPower   += (SEEK_TIME * SEEK_POWER);
            ErsRequestServiceTime       += SEEK_TIME;
            OriginalRequestServiceTime  += SEEK_TIME;
        }

        ErsPower        += ((GetChunkCapacity(video, seg, selectedVersion) / TRANSFER_TIME) * ACTIVE_POWER);
        OriginalPower   += ((GetChunkCapacity(video, seg, ver) / TRANSFER_TIME) * ACTIVE_POWER);

        ErsRequestServiceTime       += (GetChunkCapacity(video, seg, selectedVersion) / TRANSFER_TIME);
        OriginalRequestServiceTime  += (GetChunkCapacity(video, seg, ver) / TRANSFER_TIME);

        servCount++;
    }
    serviceCount += service.size();
}

int request_version_selector(){
    int rnd = Request_Version_Randnum(engine);
    for(int i = 0; i < 7; i++){
        rnd -= workloads[workload][i]*100;
        if(rnd <= 0) return i;
    }
    return -1;
}

int movieChoose(){
    int rnd = Rand(engine);
    for(int idx = 0; idx < VIDEO; idx++){
        rnd -= videoPopularity[idx]*1000000;
        if(rnd <= 0) return idx;
    }
    return -1;
}

int segmentChoose(int video){
    int rnd = Rand(engine);
    for(int idx = 0; idx < videoSegmentSize[video]; idx += SEGMENT_PARTIAL_SIZE){
        rnd -= popularitySegment[video][idx]*1000000;
        if(rnd <= 0) return idx;
    }
    return -1;
}

string GetInputFileName(){
    return to_string(END_SIMULATION) + "_" + to_string(ARRIVAL_RATE) + "_" + "Request_Input.txt";
}

string GetOutputFileName(){
    return "Disk" + to_string(DISK/1000) + "GB_Video" + to_string(VIDEO) + "_Duration" + to_string(END_SIMULATION) + "_ArrivalRate" + to_string(ARRIVAL_RATE) + "_SetDisk" + to_string(SETTING_DISK) + "_MaxDisk" + to_string(maximumDiskNumber) + "_Workload(" + to_string(workload) + ").txt";
}

int main(){
    clock_t begin, end;
    begin = clock();
    // freopen(GetOutputFileName().c_str(), "w+", stdout);
    freopen(GetInputFileName().c_str(), "r", stdin);
    // freopen("dptest.txt", "w+", stdout);

    int poissonValue;
    while (scanf("%d", &poissonValue) != EOF) poisson.push_back(poissonValue);

    ZipfDistribution(videoPopularity, VIDEO, 1 - THETA);
    for(int video = 0; video < VIDEO; video++){
        // int rnd = Video_Length_GEN(engine);
        // rnd -= rnd%(LENGTH * SEGMENT_PARTIAL_SIZE);
        // videoLength[video] = 3600 + rnd;
        videoSegmentSize[video] = videoLength[video] / LENGTH;
        uniform_int_distribution<unsigned int> segment_generator(0, VideoNum(video));
        ZipfDistributionPartialSum(popularitySegment[video], videoSegmentSize[video], SEGMENT_PARTIAL_SIZE, 1 - THETA);
        for(int seg = 0; seg < videoSegmentSize[video]; seg++) videoSegmentIdx[video].push_back(segment_generator(engine));
    }
    
    InitVideoSize();
    InitPopularity();

//    PrintPopularity();

    InitSsim();     //      ssim variation range¸¦ 1.0 ~ 5.0À¸·Î º¯È­½ÃÅ²´Ù

    GetMaximumDiskNumber();


    // minimumDiskServiceTime();
    SetNumberOfDisk(SETTING_DISK);
    Heuristic_find_X();
    // printX();

    chunk_allocation();

    printf("Disk allocated segment total cap is ");
    printCapacity(totalCapacity);

    // print_disk_seg();

    requestList = new List;
    algo3QoE.assign(NUMBER_OF_DISK, double());

    // 1000 24h 1.0 service 벡터에 푸쉬만 하는데 컴파일 178초 걸림
    // 1000 24h 1.0 QoE, Power 컴파일시 6분정도 걸림
    for(int sec = 0, client = 0; sec <= END_SIMULATION; sec++){
        requestList->Delete(sec);
        while(sec ==  poisson[client]){
            int video   = movieChoose();
            int seg     = segmentChoose(video);
            int ver     = request_version_selector();
            Request *req = new Request(client, video, seg, ver, sec, ((seg/SEGMENT_PARTIAL_SIZE)+1) * LENGTH * SEGMENT_PARTIAL_SIZE);
            requestList->Insert(req);

            client++;
        }
        // if(sec>=5000 && sec<5050) bandwidth_allocation(sec, client);
        if(sec>=5000 && sec<5002) bandwidth_allocation(sec, client);

       int ersIdleTime      = (NUMBER_OF_DISK * LENGTH - ErsRequestServiceTime);
       int originalIdleTime = (maximumDiskNumber * LENGTH - OriginalRequestServiceTime);

       if(ersIdleTime < 0) printf("ers IDLE ERROR sec: %d\n", sec);
       if(originalIdleTime < 0) printf("ori IDLE ERROR sec: %d\n", sec);

       ErsPower        += ersIdleTime * IDLE_POWER;
       OriginalPower   += originalIdleTime * IDLE_POWER;
    }

    ErsQoE      /= serviceCount;
    OriginalQoE /= serviceCount;

    ErsPower        /= END_SIMULATION;
    OriginalPower   /= END_SIMULATION;

    ErsPower += (maximumDiskNumber - NUMBER_OF_DISK) * STANDBY_POWER;

  
    printf("QoE ratio: %lf%%, Power ratio: %lf%%\n", ErsQoE/OriginalQoE*100, ErsPower/OriginalPower*100);

    // double maxPop = 0;
    // double minPop = 100;

    // for(auto p : diskSegmentPopularitySum){
    //     if(maxPop < p) maxPop = p;
    //     if(minPop > p) minPop = p;
    // }

//    printf("Maximum Disk %3.1d\tERS Disk %3.1d\n", maximumDiskNumber, NUMBER_OF_DISK);
//    printf("\nTSA ALL QoE \t\t%lf\nERS QoE \t\t%lf\n", OriginalQoE, ErsQoE);
//    printf("%lf%%\n", ErsQoE/OriginalQoE*100);
//
//    printf("\nTSA ALL Power %lfW\tERS Power %lfW\n", OriginalPower, ErsPower);
//    printf("%lf%%\n", ErsPower/OriginalPower*100);
//
//    printf("\n%d\n", changeCount);

    end = clock();

    cout<<"\n\n\nMaximum Disk: "<<maximumDiskNumber<<"\nexcution time : "<<((end-begin)/CLOCKS_PER_SEC)<<"sec"<<endl;

//    printf("Algorithm 1 Profit: %lf, Upper bound: %lf\n", al1Profit, al1Upperbound);
//
//    printf("\n\n");
//    for(int d = 0; d < NUMBER_OF_DISK; d++){
//        printf("disk : %d\t pop sum :%lf\n", d, diskSegmentPopularitySum[d]);
//    }
}


