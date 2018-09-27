// #include "config.h"
#include <cstdio>
#include <cassert>
#include <mpi.h>
#include <vector>

// #include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <infiniband/peer_ops.h>

#include "test_utils.h"

int mpi_rank = -1;

#define print(FMT, ARGS...)                                         \
do {                                                                \
    fprintf(stderr, "[%d] [%d] %s() "                               \
            FMT "\n", getpid(),  mpi_rank, __FUNCTION__ , ## ARGS); \
    fflush(stderr);                                                 \
} while(0)

struct app_context {
    int rank;
    int ib_port;
    char *buf;
    struct ibv_context *ctx;
    struct ibv_mr *mr;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibv_qp *qp;
    // struct ibv_exp_peer_commit commit;
    // struct peer_op_wr peer_op[10];
    cudaStream_t stream;
};

struct peer_info {
    uint32_t qpn;
    uint32_t rkey;
    uint16_t lid;
};

MPI_Datatype peer_info_type;

struct ibv_exp_peer_buf *peer_buf_alloc(ibv_exp_peer_buf_alloc_attr *attr);
int peer_buf_release(struct ibv_exp_peer_buf *pb);
uint64_t peer_register_va(void *start, size_t length, uint64_t peer_id, struct ibv_exp_peer_buf *pb);
int peer_unregister_va(uint64_t registration_id, uint64_t peer_id);

void create_data_type() {
    const int nitems = 2;
    int blocklengths[] = {2, 1};
    MPI_Datatype types[] = {MPI_UNSIGNED, MPI_UNSIGNED_SHORT};
    MPI_Aint offsets[nitems];
    offsets[0] = offsetof(peer_info, qpn);
    offsets[1] = offsetof(peer_info, lid);
    MPI_CHECK(MPI_Type_create_struct(nitems, blocklengths, offsets, types, &peer_info_type));
    MPI_CHECK(MPI_Type_commit(&peer_info_type));
}

void get_peer_attr(struct ibv_exp_peer_direct_attr &peer_attr,
                    const struct app_context &ctx) {
    peer_attr.peer_id = ctx.rank;
    peer_attr.buf_alloc = peer_buf_alloc;
    peer_attr.buf_release = peer_buf_release;
    peer_attr.register_va = peer_register_va;
    peer_attr.unregister_va = peer_unregister_va;
    peer_attr.caps = (IBV_EXP_PEER_OP_STORE_DWORD_CAP  | 
                    IBV_EXP_PEER_OP_STORE_QWORD_CAP    | 
                    IBV_EXP_PEER_OP_FENCE_CAP          | 
                    IBV_EXP_PEER_OP_POLL_AND_DWORD_CAP |
                    IBV_EXP_PEER_OP_POLL_NOR_DWORD_CAP |
                    IBV_EXP_PEER_OP_POLL_GEQ_DWORD_CAP |
                    IBV_EXP_PEER_OP_COPY_BLOCK_CAP     |
                    IBV_EXP_PEER_OP_STORE_QWORD_CAP    
                    );
    peer_attr.peer_dma_op_map_len = 256;
    peer_attr.comp_mask = IBV_EXP_PEER_DIRECT_VERSION;
    peer_attr.version = 1;
}

void ib_init(app_context &app_ctx) {
    int ret = 0;
    char ib_devname[] = "mlx5_2";
    struct ibv_device **dev_list;
    struct ibv_device *dev = NULL;
    struct ibv_context *ib_ctx = NULL;
    struct ibv_pd *pd = NULL;
    struct ibv_mr *mr = NULL;
    struct ibv_cq *send_cq = NULL;
    struct ibv_cq *recv_cq = NULL;
    struct ibv_qp *qp = NULL;
    int send_cq_cap = 10;
    int recv_cq_cap = 10;

    dev_list = ibv_get_device_list(NULL);
    assert(dev_list);

    // fprintf(stderr, "%p\n", dev_list);
    for (int i = 0; dev_list[i]; i++) {
        // fprintf(stderr, "%s\n", ibv_get_device_name(dev_list[i]));
        if (!strcmp(ibv_get_device_name(dev_list[i]), ib_devname)) {
            dev = dev_list[i];
            break;
        }
    }
    assert(dev);

    ib_ctx = ibv_open_device(dev);
    assert(ib_ctx);

    pd = ibv_alloc_pd(ib_ctx);
    assert(pd);

    CUDA_CHECK(cudaMalloc(&app_ctx.buf, sizeof *app_ctx.buf));
    CUDA_CHECK(cudaMemset(app_ctx.buf, 0, sizeof *app_ctx.buf));

    mr = ibv_reg_mr(pd, app_ctx.buf, sizeof *app_ctx.buf, 
                    IBV_ACCESS_LOCAL_WRITE |
                    IBV_ACCESS_REMOTE_READ |
                    IBV_ACCESS_REMOTE_WRITE);
    assert(mr);

    struct ibv_exp_peer_direct_attr peer_attr = {};
    get_peer_attr(peer_attr, app_ctx);

    struct ibv_exp_cq_init_attr cq_init_attr = {};
    cq_init_attr.comp_mask = IBV_EXP_CQ_INIT_ATTR_PEER_DIRECT;
    cq_init_attr.flags = 0;
    cq_init_attr.peer_direct_attrs = &peer_attr;

    send_cq = ibv_exp_create_cq(ib_ctx, send_cq_cap, NULL, NULL, 0, &cq_init_attr);
    assert(send_cq);

    recv_cq = ibv_exp_create_cq(ib_ctx, recv_cq_cap, NULL, NULL, 0, &cq_init_attr);
    assert(recv_cq);

    // struct ibv_qp_init_attr qp_init_attr = {
    //     NULL,
    //     send_cq,
    //     recv_cq,
    //     NULL,
    //     {1, 1, 1, 1, 0},
    //     IBV_QPT_RC,
    //     0,
    //     NULL
    // };
    // qp = ibv_create_qp(pd, &qp_init_attr);
    // assert(qp);

    struct ibv_exp_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq = send_cq;
    qp_init_attr.recv_cq = recv_cq;
    qp_init_attr.cap = {1, 1, 1, 1, 0};
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.pd = pd;
    qp_init_attr.comp_mask = IBV_EXP_QP_INIT_ATTR_PD |
                    IBV_EXP_QP_INIT_ATTR_PEER_DIRECT;
    qp_init_attr.peer_direct_attrs = &peer_attr;

    qp = ibv_exp_create_qp(ib_ctx, &qp_init_attr);
    assert(qp);

    struct ibv_qp_attr qp_attr = {};
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = app_ctx.ib_port;
    qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
				              IBV_ACCESS_REMOTE_READ |
				              IBV_ACCESS_REMOTE_WRITE;
    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX |
		        IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    ret = ibv_modify_qp(qp, &qp_attr, flags);
    assert(!ret);

    app_ctx.ctx = ib_ctx;
    app_ctx.mr = mr;
    app_ctx.send_cq = send_cq;
    app_ctx.recv_cq = recv_cq;
    app_ctx.qp = qp;
}

void peer_exch_info(const app_context &ctx, peer_info &peer) {
    int ret;
    struct ibv_port_attr port_attr;

    ret = ibv_query_port(ctx.ctx, ctx.ib_port, &port_attr);
    assert(!ret);

    peer.qpn = ctx.qp->qp_num;
    peer.rkey = ctx.mr->rkey;
    peer.lid = port_attr.lid;

    // MPI_CHECK(MPI_Sendrecv(&x, 1, MPI_INT, !ctx.rank, 0,
    //                         &y, 1, MPI_INT, !ctx.rank, 0, 
    //                         MPI_COMM_WORLD, NULL));
    MPI_CHECK(MPI_Sendrecv_replace(&peer, 1, peer_info_type, !ctx.rank, 0,
                                    !ctx.rank, 0, MPI_COMM_WORLD, NULL));
}

void qp_change_state_rtr(const app_context &ctx, const peer_info &peer) {
    int ret = 0;
    struct ibv_qp_attr qp_attr = {};
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_2048;
    qp_attr.dest_qp_num = peer.qpn;
    qp_attr.rq_psn = 0;
    qp_attr.max_dest_rd_atomic = 1;
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 0;
    qp_attr.ah_attr.dlid = peer.lid;
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = ctx.ib_port;

    ret = ibv_modify_qp(ctx.qp, &qp_attr,
                IBV_QP_STATE                |
                IBV_QP_AV                   |
                IBV_QP_PATH_MTU             |
                IBV_QP_DEST_QPN             |
                IBV_QP_RQ_PSN               |
                IBV_QP_MAX_DEST_RD_ATOMIC   |
                IBV_QP_MIN_RNR_TIMER);
    assert(!ret);
}

void qp_change_state_rts(const app_context &ctx, const peer_info &peer) {
    int ret = 0;
    struct ibv_qp_attr qp_attr = {};
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.sq_psn = 0;
    qp_attr.max_rd_atomic = 1;

    ret = ibv_modify_qp(ctx.qp, &qp_attr,
                IBV_QP_STATE            |
                IBV_QP_TIMEOUT          |
                IBV_QP_RETRY_CNT        |
                IBV_QP_RNR_RETRY        |
                IBV_QP_SQ_PSN           |
                IBV_QP_MAX_QP_RD_ATOMIC);
    assert(!ret);
}

void fill_ops(const struct ibv_exp_peer_commit &commit, 
        std::vector<CUstreamBatchMemOpParams> &gpu_ops) {
    for (int i = 0; i < commit.entries; i++) {
        struct peer_op_wr *op = &commit.storage[i];
        print("i %d type %d", i, op->type);
    }
}

void send(const app_context &ctx) {
    int ret = 0;

    struct ibv_sge sge = {
        (uint64_t) ctx.buf,
        sizeof *ctx.buf,
        ctx.mr->lkey
    };

    struct ibv_exp_send_wr wr = {};
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    // wr.opcode = IBV_WR_SEND;
    // wr.send_flags = IBV_SEND_SIGNALED;
    wr.exp_opcode = IBV_EXP_WR_SEND;
    wr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    wr.next = NULL;

    struct ibv_exp_send_wr *bad_wr;

    ret = ibv_exp_post_send(ctx.qp, &wr, &bad_wr);
    assert(!ret);

    const int num_entries = 32;
    struct peer_op_wr peer_op[num_entries] = {0}; // ?
    struct ibv_exp_peer_commit commit = {
        peer_op,
        num_entries,
        0,
        0
    };

    for (int i = 1; i < num_entries; i++) {
        peer_op[i - 1].next = &peer_op[i];
    }
    peer_op[num_entries - 1].next = NULL;

    ret = ibv_exp_peer_commit_qp(ctx.qp, &commit);
    print("%d", ret);
    assert(!ret);

    std::vector<CUstreamBatchMemOpParams> gpu_ops;

    fill_ops(commit, gpu_ops);
}

void recv(const app_context &ctx) {
    int ret = 0;

    struct ibv_sge sge = {
        (uint64_t) ctx.buf,
        sizeof *ctx.buf,
        ctx.mr->lkey
    };

    struct ibv_recv_wr wr = {
        1,
        NULL,
        &sge,
        1
    };

    struct ibv_recv_wr *bad_wr;

    ret = ibv_post_recv(ctx.qp, &wr, &bad_wr);
    assert(!ret);
}

void poll(const app_context &ctx) {
    int ne;
    const int num_entries = 10;
    struct ibv_wc wc[num_entries];
    struct ibv_cq *cq = ctx.rank ? ctx.recv_cq : ctx.send_cq;

    while (true) {
        ne = ibv_poll_cq(cq, num_entries, wc);
        assert(ne >= 0);
        if (ne > 0) {
            print("ne %d", ne);
            for (int i = 0; i < ne; i++) {
                print("status %d wr_id %d", wc[i].status, wc[i].wr_id);
            }
            break;
        }
    }
}

int main(int argc, char *argv[]) {
    int comm_size;
    app_context ctx = {};
    peer_info peer = {};

    ctx.ib_port = 1;

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));
    assert(comm_size == 2);
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank));
    mpi_rank = ctx.rank;
    create_data_type();
    gpu_init(-1);
    CUDA_CHECK(cudaStreamCreate(&ctx.stream));
    ib_init(ctx);
    print("exch");
    peer_exch_info(ctx, peer);
    print("RTR");
    qp_change_state_rtr(ctx, peer);
    print("RTS");
    qp_change_state_rts(ctx, peer);
    
    if (!ctx.rank) {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        print("send");
        CUDA_CHECK(cudaMemset(ctx.buf, 42, sizeof *ctx.buf));
        send(ctx);
    }
    else {
        print("recv");
        recv(ctx);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    print("poll");
    poll(ctx);

    if (ctx.rank) {
        char host_buf;
        CUDA_CHECK(cudaMemcpy(&host_buf, ctx.buf, sizeof *ctx.buf, cudaMemcpyDeviceToHost));
        print("buf %d", host_buf);
    }

    print("finalize");
    MPI_CHECK(MPI_Finalize());
    return 0;
}