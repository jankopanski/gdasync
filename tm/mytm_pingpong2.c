/*
 * Copyright (c) 2017 Mellanox Technologies, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#define _GNU_SOURCE
#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <time.h>

#include "pingpong.h"

#define WR_ID(x) (1 << (x))
#define RECV_WR_ID(x) (1 << (x + SCENARIO_MAX))
#define ADD_WR_ID(x) (1 << (x + SCENARIO_MAX * 2))
#define FIN_WR_ID(x) (1 << (x + SCENARIO_MAX * 3))

enum {
	SCENARIO_UNEXP,
	SCENARIO_EAGER,
	SCENARIO_RNDV,
	SCENARIO_MAX = 4,

	WR_ID_SCENARIO_EAGER = WR_ID(SCENARIO_EAGER) |
			  RECV_WR_ID(SCENARIO_EAGER),
	WR_ID_SCENARIO_RNDV  = WR_ID(SCENARIO_RNDV)  |
			  RECV_WR_ID(SCENARIO_RNDV),
	WR_ID_SCENARIO_UNEXP = WR_ID(SCENARIO_UNEXP) |
			  RECV_WR_ID(SCENARIO_UNEXP),
	WR_ID_SCENARIO_ALL   = WR_ID_SCENARIO_EAGER  |
			       WR_ID_SCENARIO_RNDV   |
			       WR_ID_SCENARIO_UNEXP,
	WR_ID_SCENARIO_RECV  =
			  RECV_WR_ID(SCENARIO_EAGER) |
			  RECV_WR_ID(SCENARIO_RNDV)  |
			  RECV_WR_ID(SCENARIO_UNEXP),
	WR_ID_SCENARIO_TAG  = WR_ID(SCENARIO_EAGER)  |
			      WR_ID(SCENARIO_RNDV),

	TAG_MASK	 = 0xffffUL,
};

static int page_size;

struct pingpong_context {
	struct ibv_context	*context;
	struct ibv_pd		*pd;
	struct ibv_mr		*mr;
	struct ibv_cq		*cq;
	struct ibv_srq		*srq;
	struct ibv_qp		*send_qp;
	void			*buf;
	int			 size;
	int			 buf_size;
	int			 send_flags;
	int			 pending;
	struct ibv_port_attr	 portinfo;
	int			 unexp_cnt;
};

struct pingpong_dest {
	int lid;
	int send_qpn;
	int psn;
	union ibv_gid gid;
};

static int pp_connect_ctx(struct pingpong_context *ctx, int port, int my_psn,
			  enum ibv_mtu mtu, int sl,
			  struct pingpong_dest *dest, int sgid_idx)
{
	int flags;
	struct ibv_qp_attr attr = {
		.qp_state		= IBV_QPS_RTR,
		.path_mtu		= mtu,
		.dest_qp_num		= dest->send_qpn,
		.rq_psn			= dest->psn,
		.max_dest_rd_atomic	= 1,
		.min_rnr_timer		= 12,
		.ah_attr		= {
			.is_global	= 0,
			.dlid		= dest->lid,
			.sl		= sl,
			.src_path_bits	= 0,
			.port_num	= port
		}
	};

	if (dest->gid.global.interface_id) {
		attr.ah_attr.is_global = 1;
		attr.ah_attr.grh.hop_limit = 1;
		attr.ah_attr.grh.dgid = dest->gid;
		attr.ah_attr.grh.sgid_index = sgid_idx;
	}

	flags = IBV_QP_STATE              |
		IBV_QP_AV                 |
		IBV_QP_PATH_MTU           |
		IBV_QP_DEST_QPN           |
		IBV_QP_RQ_PSN             |
		IBV_QP_MAX_DEST_RD_ATOMIC |
		IBV_QP_MIN_RNR_TIMER;

	if (ibv_modify_qp(ctx->send_qp, &attr, flags)) {
		fprintf(stderr, "Failed to modify send QP to RTR\n");
		return 1;
	}

	flags = IBV_QP_STATE              |
		IBV_QP_TIMEOUT            |
		IBV_QP_RETRY_CNT          |
		IBV_QP_RNR_RETRY          |
		IBV_QP_SQ_PSN             |
		IBV_QP_MAX_QP_RD_ATOMIC;


	attr.qp_state	    = IBV_QPS_RTS;
	attr.timeout	    = 14;
	attr.retry_cnt	    = 7;
	attr.rnr_retry	    = 7;
	attr.sq_psn	    = my_psn;
	attr.max_rd_atomic  = 1;

	if (ibv_modify_qp(ctx->send_qp, &attr, flags)) {
		fprintf(stderr, "Failed to modify send QP to RTS\n");
		return 1;
	}

	return 0;
}

static struct pingpong_dest *pp_client_exch_dest(const char *servername,
						 int port,
						 struct pingpong_dest *my_dest)
{
	struct addrinfo *res, *t;
	struct addrinfo hints = {
		.ai_family   = AF_UNSPEC,
		.ai_socktype = SOCK_STREAM
	};
	char *service;
	char msg[sizeof "0000:000000:00000000000000000000000000000000"];
	int n;
	int sockfd = -1;
	struct pingpong_dest *rem_dest = NULL;
	char gid[33];

	if (asprintf(&service, "%d", port) < 0)
		return NULL;

	n = getaddrinfo(servername, service, &hints, &res);

	if (n < 0) {
		fprintf(stderr, "%s for %s:%d\n", gai_strerror(n),
			servername, port);
		free(service);
		return NULL;
	}

	for (t = res; t; t = t->ai_next) {
		sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
		if (sockfd >= 0) {
			if (!connect(sockfd, t->ai_addr, t->ai_addrlen))
				break;
			close(sockfd);
			sockfd = -1;
		}
	}

	freeaddrinfo(res);
	free(service);

	if (sockfd < 0) {
		fprintf(stderr, "Couldn't connect to %s:%d\n",
			servername, port);
		return NULL;
	}

	gid_to_wire_gid(&my_dest->gid, gid);
	sprintf(msg, "%04x:%06x:%06x:%s", my_dest->lid, my_dest->send_qpn, my_dest->psn, gid);
	if (write(sockfd, msg, sizeof(msg)) != sizeof(msg)) {
		fprintf(stderr, "Couldn't send local address\n");
		goto out;
	}

	int wn, rn;
	if ((rn = read(sockfd, msg, sizeof(msg))) != sizeof(msg) ||
	    (wn = write(sockfd, "done", sizeof "done")) != sizeof "done") {
		perror("client read/write");
		fprintf(stderr, "Couldn't read/write remote address\n");
		fprintf(stderr, "%d %d\n", wn, rn);
		goto out;
	}

	rem_dest = malloc(sizeof(*rem_dest));
	if (!rem_dest)
		goto out;

	if (sscanf(msg, "%x:%x:%x:%s", &rem_dest->lid,
					  &rem_dest->send_qpn,
					  &rem_dest->psn, gid) != 4) {
		fprintf(stderr, "Couldn't parse remote address\n");
	}
	wire_gid_to_gid(gid, &rem_dest->gid);

out:
	close(sockfd);
	return rem_dest;
}

static struct pingpong_dest *pp_server_exch_dest(struct pingpong_context *ctx,
						 int ib_port, enum ibv_mtu mtu,
						 int port, int sl,
						 struct pingpong_dest *my_dest,
						 int sgid_idx)
{
	struct addrinfo *res, *t;
	struct addrinfo hints = {
		.ai_flags    = AI_PASSIVE,
		.ai_family   = AF_UNSPEC,
		.ai_socktype = SOCK_STREAM
	};
	char *service;
	char msg[sizeof "0000:000000:00000000000000000000000000000000"];
	int n;
	int sockfd = -1, connfd;
	struct pingpong_dest *rem_dest = NULL;
	char gid[33];

	if (asprintf(&service, "%d", port) < 0)
		return NULL;

	n = getaddrinfo(NULL, service, &hints, &res);

	if (n < 0) {
		fprintf(stderr, "%s for port %d\n", gai_strerror(n), port);
		free(service);
		return NULL;
	}

	for (t = res; t; t = t->ai_next) {
		sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
		if (sockfd >= 0) {
			n = 1;

			setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &n,
				   sizeof(n));

			if (!bind(sockfd, t->ai_addr, t->ai_addrlen))
				break;
			close(sockfd);
			sockfd = -1;
		}
	}

	freeaddrinfo(res);
	free(service);

	if (sockfd < 0) {
		fprintf(stderr, "Couldn't listen to port %d\n", port);
		return NULL;
	}

	listen(sockfd, 1);
	connfd = accept(sockfd, NULL, NULL);
	close(sockfd);
	if (connfd < 0) {
		fprintf(stderr, "accept() failed\n");
		return NULL;
	}

	n = read(connfd, msg, sizeof(msg));
	if (n != sizeof(msg)) {
		perror("server read");
		fprintf(stderr, "%d/%zd: Couldn't read remote address\n", n,
			sizeof(msg));
		goto out;
	}

	rem_dest = malloc(sizeof(*rem_dest));
	if (!rem_dest)
		goto out;

	if (sscanf(msg, "%x:%x:%x:%s", &rem_dest->lid,
					  &rem_dest->send_qpn,
					  &rem_dest->psn, gid) != 4) {
		fprintf(stderr, "Couldn't parse remote address\n");
	}

	wire_gid_to_gid(gid, &rem_dest->gid);

	if (pp_connect_ctx(ctx, ib_port, my_dest->psn, mtu, sl, rem_dest,
								sgid_idx)) {
		fprintf(stderr, "Couldn't connect to remote QP\n");
		free(rem_dest);
		rem_dest = NULL;
		goto out;
	}


	gid_to_wire_gid(&my_dest->gid, gid);
	sprintf(msg, "%04x:%06x:%06x:%s", my_dest->lid, my_dest->send_qpn, my_dest->psn, gid);
	fprintf(stderr, "%s_%d\n", msg, sizeof(msg));
	int wn, rn;
	if ((wn = write(connfd, msg, sizeof(msg))) != sizeof(msg) ||
	    (rn = read(connfd, msg, sizeof(msg))) != sizeof "done") {
		fprintf(stderr, "Couldn't send/recv local address\n");
		fprintf(stderr, "%d %d\n", wn, rn);
		free(rem_dest);
		rem_dest = NULL;
		goto out;
	}


out:
	close(connfd);
	return rem_dest;
}

static struct pingpong_context *pp_init_ctx(struct ibv_device *ib_dev, int size,
					    int port)
{
	struct ibv_exp_device_attr device_attr;
	struct pingpong_context *ctx;
	int total_buf_size, flags;

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->size       = size;
	ctx->send_flags = IBV_SEND_SIGNALED;
	ctx->buf_size   = (size + page_size - 1) / page_size * page_size;

	total_buf_size = ctx->buf_size * SCENARIO_MAX * 2;

	ctx->buf = memalign(page_size, total_buf_size);
	if (!ctx->buf) {
		fprintf(stderr, "Couldn't allocate work buf.\n");
		goto clean_ctx;
	}

	memset(ctx->buf, 0, total_buf_size);

	ctx->context = ibv_open_device(ib_dev);
	if (!ctx->context) {
		fprintf(stderr, "Couldn't get context for %s\n",
			ibv_get_device_name(ib_dev));
		goto clean_buffer;
	}

	memset(&device_attr, 0, sizeof(device_attr));
	device_attr.comp_mask = IBV_EXP_DEVICE_ATTR_RESERVED - 1;
	if (ibv_exp_query_device(ctx->context, &device_attr)) {
		fprintf(stderr, "Couldn't query device\n");
		goto clean_device;
	}

	if (!device_attr.tm_caps.max_num_tags ||
	    !(device_attr.tm_caps.capability_flags & IBV_EXP_TM_CAP_RC)) {
		fprintf(stderr, "Tag matching not supported\n");
		goto clean_device;
	}

	ctx->pd = ibv_alloc_pd(ctx->context);
	if (!ctx->pd) {
		fprintf(stderr, "Couldn't allocate PD\n");
		goto clean_device;
	}

	ctx->mr = ibv_reg_mr(ctx->pd, ctx->buf, total_buf_size,
			     IBV_ACCESS_LOCAL_WRITE |
			     IBV_ACCESS_REMOTE_READ |
			     IBV_ACCESS_REMOTE_WRITE);
	if (!ctx->mr) {
		fprintf(stderr, "Couldn't register MR\n");
		goto clean_pd;
	}

	ctx->cq = ibv_create_cq(ctx->context, SCENARIO_MAX * 3, NULL, NULL, 0);

	if (!ctx->cq) {
		fprintf(stderr, "Couldn't create CQ\n");
		goto clean_mr;
	}

	struct ibv_exp_create_srq_attr attr = {
		.base.attr = {
			.max_wr  = 33,
			.max_sge = 1
		},
		.comp_mask =
			IBV_EXP_CREATE_SRQ_CQ |
			IBV_EXP_CREATE_SRQ_TM,
		.srq_type = IBV_EXP_SRQT_TAG_MATCHING,
		.pd = ctx->pd,
		.cq = ctx->cq,
		.tm_cap = {
			.max_num_tags = 10,
			.max_ops = 10,
		}
	};

	ctx->srq = ibv_exp_create_srq(ctx->context, &attr);

	if (!ctx->srq)  {
		fprintf(stderr, "Couldn't create SRQ\n");
		goto clean_cq;
	}

	struct ibv_qp_init_attr init_attr = {
		.send_cq = ctx->cq,
		.recv_cq = ctx->cq,
		.cap     = {
			.max_send_wr  = SCENARIO_MAX,
			.max_send_sge = 1,
			.max_recv_wr  = 1,
			.max_recv_sge = 1,
		},
		.qp_type = IBV_QPT_RC
	};

	init_attr.srq = ctx->srq;
	// init_attr.cap.max_send_wr = 0;
	// init_attr.cap.max_recv_wr = 0;

	ctx->send_qp = ibv_create_qp(ctx->pd, &init_attr);
	if (!ctx->send_qp)  {
		fprintf(stderr, "Couldn't create send QP\n");
		goto clean_srq;
	}

	struct ibv_qp_attr qp_attr = {
		.qp_state        = IBV_QPS_INIT,
		.pkey_index	 = 0,
		.port_num	 = port,
		.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
				   IBV_ACCESS_REMOTE_READ |
				   IBV_ACCESS_REMOTE_WRITE,
	};

	flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX |
		IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

	if (ibv_modify_qp(ctx->send_qp, &qp_attr, flags)) {
		fprintf(stderr, "Failed to modify send QP to INIT\n");
		goto clean_qps;
	}

	return ctx;

clean_qps:
clean_send_qp:
	ibv_destroy_qp(ctx->send_qp);

clean_srq:
	ibv_destroy_srq(ctx->srq);

clean_cq:
	ibv_destroy_cq(ctx->cq);

clean_mr:
	ibv_dereg_mr(ctx->mr);

clean_pd:
	ibv_dealloc_pd(ctx->pd);

clean_device:
	ibv_close_device(ctx->context);

clean_buffer:
	free(ctx->buf);

clean_ctx:
	free(ctx);

	return NULL;
}

static int pp_close_ctx(struct pingpong_context *ctx)
{
	if (ibv_destroy_qp(ctx->send_qp)) {
		fprintf(stderr, "Couldn't destroy send QP\n");
		return 1;
	}

	if (ibv_destroy_srq(ctx->srq)) {
		fprintf(stderr, "Couldn't destroy SRQ\n");
		return 1;
	}

	if (ibv_destroy_cq(ctx->cq)) {
		fprintf(stderr, "Couldn't destroy CQ\n");
		return 1;
	}

	if (ibv_dereg_mr(ctx->mr)) {
		fprintf(stderr, "Couldn't deregister MR\n");
		return 1;
	}

	if (ibv_dealloc_pd(ctx->pd)) {
		fprintf(stderr, "Couldn't deallocate PD\n");
		return 1;
	}

	if (ibv_close_device(ctx->context)) {
		fprintf(stderr, "Couldn't release context\n");
		return 1;
	}

	free(ctx->buf);
	free(ctx);

	return 0;
}

static void pp_post_recv(struct pingpong_context *ctx, int n)
{
	struct ibv_sge list = {
		.addr	= (uintptr_t) ctx->buf,
		.length = ctx->size,
		.lkey	= ctx->mr->lkey
	};
	struct ibv_recv_wr wr = {
		.wr_id	    = RECV_WR_ID(SCENARIO_UNEXP),
		.sg_list    = &list,
		.num_sge    = 1,
	};
	struct ibv_recv_wr *bad_wr;
	int i;

	for (i = 0; i < n; ++i) {
		if (ibv_post_srq_recv(ctx->srq, &wr, &bad_wr)) {
			fprintf(stderr, "Couldn't post receive\n");
			exit(-1);
		}
	}
}

static void pp_post_recv_tm(struct pingpong_context *ctx, int tag)
{
	void *buf = ctx->buf + ctx->buf_size * tag;
	struct ibv_sge list = {
		.addr	= (uintptr_t)buf,
		.length = ctx->size,
		.lkey	= ctx->mr->lkey
	};
	struct ibv_exp_ops_wr wr = {
		.wr_id	    = ADD_WR_ID(tag),
		.opcode	    = IBV_EXP_WR_TAG_ADD,
		.flags	    = IBV_EXP_OPS_SIGNALED | IBV_EXP_OPS_TM_SYNC,
		.tm	    = {
			.unexpected_cnt = ctx->unexp_cnt,
			.add = {
				.recv_wr_id  = RECV_WR_ID(tag),
				.sg_list     = &list,
				.num_sge     = 1,
				.tag	     = tag,
				.mask	     = TAG_MASK
			}
		}
	};
	struct ibv_exp_ops_wr *bad_wr;

	ctx->pending |= ADD_WR_ID(tag);

	if (ibv_exp_post_srq_ops(ctx->srq, &wr, &bad_wr)) {
		fprintf(stderr, "Couldn't post receive TM %x\n", tag);
		exit(-1);
	}
}

static void pp_post_sync(struct pingpong_context *ctx)
{
	struct ibv_exp_ops_wr wr = {
		.opcode	    = IBV_EXP_WR_TAG_SYNC,
		.flags	    = IBV_EXP_OPS_TM_SYNC,
		.tm	    = {
			.unexpected_cnt = ctx->unexp_cnt,
		}
	};
	struct ibv_exp_ops_wr *bad_wr;

	if (ibv_exp_post_srq_ops(ctx->srq, &wr, &bad_wr)) {
		fprintf(stderr, "Couldn't post sync TM\n");
		exit(-1);
	}
}

static void pp_post_send_tm(struct pingpong_context *ctx, int tag)
{
	void *buf = ctx->buf + ctx->buf_size * (tag + SCENARIO_MAX);
	struct ibv_sge list = {
		.addr	= (uintptr_t)buf,
		.length = ctx->size,
		.lkey	= ctx->mr->lkey
	};
	struct ibv_send_wr wr = {
		.wr_id	    = WR_ID(tag),
		.sg_list    = &list,
		.num_sge    = 1,
		.opcode     = IBV_WR_SEND,
		.send_flags = ctx->send_flags,
	};
	struct ibv_send_wr *bad_wr;
	struct ibv_exp_tmh *tmh = buf;

	tmh->opcode = IBV_EXP_TMH_EAGER;
	tmh->tag = htobe64(tag);

	int ret;
	if (ret = ibv_post_send(ctx->send_qp, &wr, &bad_wr)) {
		fprintf(stderr, "Couldn't post send TM %x %d\n", tag, ret);
		exit(-1);
	}
}

struct pp_rvh {
	struct ibv_exp_tmh tmh;
	struct ibv_exp_tmh_rvh rvh;
};

static void pp_post_send_rndv(struct pingpong_context *ctx)
{
	int tag = SCENARIO_RNDV;
	void *buf = ctx->buf + ctx->buf_size * (tag + SCENARIO_MAX);
	struct pp_rvh *rvh = buf;
	struct ibv_sge list = {
		.addr	= (uintptr_t)buf,
		.length = sizeof(*rvh),
		.lkey	= ctx->mr->lkey
	};
	struct ibv_send_wr wr = {
		.wr_id	    = WR_ID(tag),
		.sg_list    = &list,
		.num_sge    = 1,
		.opcode     = IBV_WR_SEND,
		.send_flags = ctx->send_flags,
	};
	struct ibv_recv_wr rwr = {
		.wr_id	    = FIN_WR_ID(SCENARIO_RNDV),
		.sg_list    = &list,
		.num_sge    = 1,
	};
	struct ibv_send_wr *bad_wr;
	struct ibv_recv_wr *bad_rwr;


	rvh->tmh.opcode = IBV_EXP_TMH_RNDV;
	rvh->tmh.tag = htobe64(tag);

	rvh->rvh.rkey = htobe32(ctx->mr->rkey);
	rvh->rvh.va = htobe64((uintptr_t)buf);
	rvh->rvh.len = htobe32(ctx->size);

	ctx->pending |= FIN_WR_ID(SCENARIO_RNDV);
	if (ibv_post_recv(ctx->send_qp, &rwr, &bad_rwr)) {
		fprintf(stderr, "Couldn't post receive for FIN\n");
		exit(-1);
	}

	if (ibv_post_send(ctx->send_qp, &wr, &bad_wr)) {
		fprintf(stderr, "Couldn't post send TM %x\n", tag);
		exit(-1);
	}
}

static int pp_poll_cq(struct pingpong_context *ctx, int n,
		      struct ibv_exp_wc *wc)
{
	return ibv_exp_poll_cq(ctx->cq, n, wc, sizeof(*wc));
}

static void usage(const char *argv0)
{
	printf("Usage:\n");
	printf("  %s          start a server and wait for connection\n", argv0);
	printf("  %s <host>   connect to server at <host>\n", argv0);
	printf("\n");
	printf("Options:\n");
	printf("  -p, --port=<port>      listen on/connect to port <port> (default 18515)\n");
	printf("  -d, --ib-dev=<dev>     use IB device <dev> (default first device found)\n");
	printf("  -i, --ib-port=<port>   use port <port> of IB device (default 1)\n");
	printf("  -s, --size=<size>      size of message to exchange (default 4096)\n");
	printf("  -m, --mtu=<size>       path MTU (default 1024)\n");
	printf("  -n, --iters=<iters>    number of exchanges per QP(default 1000)\n");
	printf("  -l, --sl=<sl>          service level value\n");
	printf("  -g, --gid-idx=<gid>    local port gid index\n");
	printf("  -t  --scenario=<all|eager|rndv|unexp>\n");
	printf("  -v  --verbose		 verbose output\n");
}

int main(int argc, char *argv[])
{
	struct ibv_device      **dev_list;
	struct ibv_device	*ib_dev;
	struct ibv_exp_wc	*wc;
	struct pingpong_context *ctx;
	struct pingpong_dest     my_dest;
	struct pingpong_dest    *rem_dest;
	struct timeval           start, end;
	char                    *ib_devname = NULL;
	char                    *servername = NULL;
	unsigned int             port = 18515;
	int                      ib_port = 1;
	unsigned int             size = 4096;
	enum ibv_mtu		 mtu = IBV_MTU_1024;
	unsigned int             iters = 1000;
	int                      cnt = 0;
	int			 num_wc;
	int                      i;
	int                      sl = 0;
	int			 gidx = -1;
	char			 gid[33];
	int			 ne;
	int			 scenario = WR_ID_SCENARIO_ALL;
	int			 scen_num;
	int			 verbose = 0;

	srand48(getpid() * time(NULL));

	while (1) {
		int c;

		static struct option long_options[] = {
			{ .name = "port",	  .has_arg = 1, .val = 'p' },
			{ .name = "ib-dev",	  .has_arg = 1, .val = 'd' },
			{ .name = "ib-port",	  .has_arg = 1, .val = 'i' },
			{ .name = "size",	  .has_arg = 1, .val = 's' },
			{ .name = "mtu",	  .has_arg = 1, .val = 'm' },
			{ .name = "iters",	  .has_arg = 1, .val = 'n' },
			{ .name = "sl",		  .has_arg = 1, .val = 'l' },
			{ .name = "gid-idx",	  .has_arg = 1, .val = 'g' },
			{ .name = "scenario",	  .has_arg = 1, .val = 't' },
			{ .name = "verbose",	  .has_arg = 0, .val = 'v' },
			{}
		};

		c = getopt_long(argc, argv, "p:d:i:s:m:r:n:l:g:t:v",
				long_options, NULL);
		if (c == -1)
			break;

		switch (c) {
		case 'p':
			port = strtoul(optarg, NULL, 0);
			if (port > 65535) {
				usage(argv[0]);
				return 1;
			}
			break;

		case 'd':
			ib_devname = strdupa(optarg);
			break;

		case 'i':
			ib_port = strtol(optarg, NULL, 0);
			if (ib_port < 1) {
				usage(argv[0]);
				return 1;
			}
			break;

		case 's':
			size = strtoul(optarg, NULL, 0);
			if (size < 1) {
				usage(argv[0]);
				return 1;
			}
			break;

		case 'm':
			mtu = pp_mtu_to_enum(strtol(optarg, NULL, 0));
			if (mtu == 0) {
				usage(argv[0]);
				return 1;
			}
			break;

		case 'n':
			iters = strtoul(optarg, NULL, 0);
			break;

		case 'l':
			sl = strtol(optarg, NULL, 0);
			break;

		case 'g':
			gidx = strtol(optarg, NULL, 0);
			break;

		case 'v':
			verbose++;
			break;

		case 't':
			scenario = 0;
			if (strstr(optarg, "eager"))
				scenario |= WR_ID_SCENARIO_EAGER;
			if (strstr(optarg, "rndv"))
				scenario |= WR_ID_SCENARIO_RNDV;
			if (strstr(optarg, "unexp"))
				scenario |= WR_ID_SCENARIO_UNEXP;
			if (strstr(optarg, "all"))
				scenario |= WR_ID_SCENARIO_ALL;

			break;

		default:
			usage(argv[0]);
			return 1;
		}
	}

	if (optind == argc - 1)
		servername = strdupa(argv[optind]);
	else if (optind < argc) {
		usage(argv[0]);
		return 1;
	}

	scen_num = 0;
	if (scenario & WR_ID(SCENARIO_EAGER))
		scen_num++;
	if (scenario & WR_ID(SCENARIO_RNDV))
		scen_num++;
	if (scenario & WR_ID(SCENARIO_UNEXP))
		scen_num++;

	num_wc = SCENARIO_MAX * 3;
	wc     = alloca(num_wc * sizeof(*wc));

	page_size = sysconf(_SC_PAGESIZE);

	dev_list = ibv_get_device_list(NULL);
	if (!dev_list) {
		perror("Failed to get IB devices list");
		return 1;
	}

	if (!ib_devname) {
		ib_dev = *dev_list;
		if (!ib_dev) {
			fprintf(stderr, "No IB devices found\n");
			return 1;
		}
	} else {
		for (i = 0; dev_list[i]; ++i)
			if (!strcmp(ibv_get_device_name(dev_list[i]),
				    ib_devname))
				break;
		ib_dev = dev_list[i];
		if (!ib_dev) {
			fprintf(stderr, "IB device %s not found\n", ib_devname);
			return 1;
		}
	}

	ctx = pp_init_ctx(ib_dev, size, ib_port);
	if (!ctx)
		return 1;

	pp_post_recv(ctx, 33);

	if (scenario & WR_ID(SCENARIO_EAGER))
		pp_post_recv_tm(ctx, SCENARIO_EAGER);
	if (scenario & WR_ID(SCENARIO_RNDV))
		pp_post_recv_tm(ctx, SCENARIO_RNDV);

	memset(&my_dest, 0, sizeof(my_dest));

	if (pp_get_port_info(ctx->context, ib_port, &ctx->portinfo)) {
		fprintf(stderr, "Couldn't get port info\n");
		return 1;
	}
	my_dest.send_qpn = ctx->send_qp->qp_num;
	my_dest.psn = lrand48() & 0xffffff;
	my_dest.lid = ctx->portinfo.lid;
	if (ctx->portinfo.link_layer != IBV_LINK_LAYER_ETHERNET
						&& !my_dest.lid) {
		fprintf(stderr, "Couldn't get local LID\n");
		return 1;
	}

	if (gidx >= 0) {
		if (ibv_query_gid(ctx->context, ib_port, gidx,
						&my_dest.gid)) {
			fprintf(stderr, "Could not get local gid for "
						"gid index %d\n", gidx);
			return 1;
		}
	} else
		memset(&my_dest.gid, 0, sizeof(my_dest.gid));

	inet_ntop(AF_INET6, &my_dest.gid, gid, sizeof(gid));
	printf("  local address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x, "
		"GID %s\n", my_dest.lid, my_dest.send_qpn,
		my_dest.psn, gid);

	if (servername)
		rem_dest = pp_client_exch_dest(servername, port, &my_dest);
	else
		rem_dest = pp_server_exch_dest(ctx, ib_port, mtu, port, sl,
								&my_dest, gidx);

	if (!rem_dest)
		return 1;

	inet_ntop(AF_INET6, &rem_dest->gid, gid, sizeof(gid));
	printf("  remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, "
		"GID %s\n", rem_dest->lid, rem_dest->send_qpn, rem_dest->psn, gid);

	if (servername)
		if (pp_connect_ctx(ctx, ib_port, my_dest.psn, mtu, sl,
				   rem_dest, gidx))
			return 1;

	if (servername) {
		if (scenario & WR_ID(SCENARIO_EAGER))
			pp_post_send_tm(ctx, SCENARIO_EAGER);
		if (scenario & WR_ID(SCENARIO_RNDV))
			pp_post_send_rndv(ctx);
		if (scenario & WR_ID(SCENARIO_UNEXP))
			pp_post_send_tm(ctx, SCENARIO_UNEXP);

		ctx->pending |= scenario;
	} else {
		ctx->pending |= scenario & WR_ID_SCENARIO_RECV;
	}

	if (gettimeofday(&start, NULL)) {
		perror("gettimeofday");
		return 1;
	}

	while (cnt < iters) {
		do {
			ne = pp_poll_cq(ctx, num_wc, wc);
			if (ne < 0) {
				fprintf(stderr, "poll CQ failed %d\n", ne);
				return 1;
			}
		} while (ne < 1);

		for (i = 0; i < ne; ++i) {
			if (wc[i].status != IBV_WC_SUCCESS) {
				fprintf(stderr, "Failed status %s (%d) for wr_id %lx\n",
					ibv_wc_status_str(wc[i].status),
					wc[i].status, wc[i].wr_id);
				return 1;
			}

			if (verbose) {
				printf("%d poll status %d (%d) %d for wr_id %lx %lx %x\n",
				       cnt, getpid(), wc[i].status,
				       wc[i].exp_opcode, wc[i].wr_id,
				       wc[i].exp_wc_flags, ctx->pending);
			}

			if (!(ctx->pending & wc[i].wr_id)) {
				fprintf(stderr, "Completion for unexpected wr_id %lx\n",
					wc[i].wr_id);
				return 1;
			}

			switch (wc[i].wr_id) {
			case RECV_WR_ID(SCENARIO_EAGER):
				if (wc[i].exp_wc_flags & IBV_EXP_WC_TM_MATCH)
					pp_post_recv_tm(ctx, SCENARIO_EAGER);
				if (!(wc[i].exp_wc_flags & IBV_EXP_WC_TM_DATA_VALID))
					continue;
				break;

			case RECV_WR_ID(SCENARIO_RNDV):
				if (wc[i].exp_wc_flags & IBV_EXP_WC_TM_MATCH)
					pp_post_recv_tm(ctx, SCENARIO_RNDV);
				if (!(wc[i].exp_wc_flags & IBV_EXP_WC_TM_DATA_VALID))
					continue;
				break;

			case RECV_WR_ID(SCENARIO_UNEXP):
				pp_post_recv(ctx, 1);
				ctx->unexp_cnt++;
				if (scenario & WR_ID_SCENARIO_TAG)
					pp_post_sync(ctx);
				break;

			case ADD_WR_ID(SCENARIO_EAGER):
			case ADD_WR_ID(SCENARIO_RNDV):
				if (wc[i].exp_wc_flags & IBV_EXP_WC_TM_SYNC_REQ)
					pp_post_sync(ctx);
				break;
			}

			ctx->pending &= ~wc[i].wr_id;
			if (!ctx->pending) {
				cnt++;

				ctx->pending = scenario;
				if (scenario & WR_ID(SCENARIO_EAGER))
					pp_post_send_tm(ctx, SCENARIO_EAGER);
				if (scenario & WR_ID(SCENARIO_RNDV))
					pp_post_send_rndv(ctx);
				if (scenario & WR_ID(SCENARIO_UNEXP))
					pp_post_send_tm(ctx, SCENARIO_UNEXP);
			}

		}
	}

	if (gettimeofday(&end, NULL)) {
		perror("gettimeofday");
		return 1;
	}

	{
		float usec = (end.tv_sec - start.tv_sec) * 1000000 +
			(end.tv_usec - start.tv_usec);
		long long bytes = (long long) size * iters * 2 * scen_num;

		printf("%lld bytes in %.2f seconds = %.2f Mbit/sec\n",
		       bytes, usec / 1000000., bytes * 8. / usec);
		printf("%d iters in %.2f seconds = %.2f usec/iter\n",
		       iters, usec / 1000000., usec / iters);
	}

	if (pp_close_ctx(ctx))
		return 1;

	ibv_free_device_list(dev_list);
	free(rem_dest);

	return 0;
}
