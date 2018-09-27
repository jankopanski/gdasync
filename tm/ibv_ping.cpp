#include <cstdio>
#include <cassert>
#include <mpi.h>

#include <infiniband/verbs.h>

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
};

struct peer_info {
    uint32_t qpn;
    uint32_t rkey;
    uint16_t lid;
};

MPI_Datatype peer_info_type;

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

    send_cq = ibv_create_cq(ib_ctx, send_cq_cap, NULL, NULL, 0);
    assert(send_cq);

    recv_cq = ibv_create_cq(ib_ctx, recv_cq_cap, NULL, NULL, 0);
    assert(send_cq);

    struct ibv_qp_init_attr qp_init_attr = {
        NULL,
        send_cq,
        recv_cq,
        NULL,
        {1, 1, 1, 1, 0},
        IBV_QPT_RC,
        0,
        NULL
    };

    qp = ibv_create_qp(pd, &qp_init_attr);
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

void send(const app_context &ctx) {
    int ret = 0;

    struct ibv_sge sge = {
        (uint64_t) ctx.buf,
        sizeof *ctx.buf,
        ctx.mr->lkey
    };

    struct ibv_send_wr wr = {};
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.next = NULL;

    struct ibv_send_wr *bad_wr;

    ret = ibv_post_send(ctx.qp, &wr, &bad_wr);
    assert(!ret);
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