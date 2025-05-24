import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { invoke } from '@tauri-apps/api/tauri';
import { 
  Card, 
  Typography, 
  Button, 
  Space, 
  Table, 
  Tag, 
  Statistic, 
  Row, 
  Col,
  Progress,
  Empty
} from 'antd';
import {
  ExperimentOutlined,
  RocketOutlined,
  LineChartOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import '../styles/Dashboard.css';

const { Title, Text } = Typography;

function Dashboard({ experiments }) {
  const [stats, setStats] = useState({
    total: 0,
    running: 0,
    completed: 0,
    bestReward: null
  });

  useEffect(() => {
    if (experiments && experiments.length > 0) {
      const running = experiments.filter(e => e.status === 'running').length;
      const completed = experiments.filter(e => e.status === 'completed').length;
      
      // Find best performing model based on reward gap
      let bestReward = null;
      let bestRewardValue = -Infinity;
      
      experiments.forEach(exp => {
        if (exp.metrics && exp.metrics.reward_gap > bestRewardValue) {
          bestRewardValue = exp.metrics.reward_gap;
          bestReward = {
            name: exp.name,
            value: exp.metrics.reward_gap,
            id: exp.id
          };
        }
      });
      
      setStats({
        total: experiments.length,
        running,
        completed,
        bestReward
      });
    }
  }, [experiments]);

  const getStatusTag = (status) => {
    switch(status) {
      case 'created':
        return <Tag icon={<ClockCircleOutlined />} color="default">Created</Tag>;
      case 'running':
        return <Tag icon={<SyncOutlined spin />} color="processing">Running</Tag>;
      case 'completed':
        return <Tag icon={<CheckCircleOutlined />} color="success">Completed</Tag>;
      case 'failed':
        return <Tag color="error">Failed</Tag>;
      default:
        return <Tag color="default">{status}</Tag>;
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => <Link to={`/experiment/${record.id}`}>{text}</Link>,
    },
    {
      title: 'Model',
      dataIndex: 'model_name',
      key: 'model_name',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: status => getStatusTag(status),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: timestamp => new Date(timestamp).toLocaleString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space size="small">
          <Button type="primary" size="small">
            <Link to={`/experiment/${record.id}`}>View</Link>
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <Title level={2}>Reward Modeling Dashboard</Title>
        <Button type="primary" icon={<ExperimentOutlined />}>
          <Link to="/new-experiment">New Experiment</Link>
        </Button>
      </div>

      <Row gutter={16} className="stats-cards">
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Experiments"
              value={stats.total}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Running"
              value={stats.running}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Completed"
              value={stats.completed}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Best Reward Gap"
              value={stats.bestReward ? stats.bestReward.value.toFixed(2) : "N/A"}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
            {stats.bestReward && (
              <Text type="secondary">{stats.bestReward.name}</Text>
            )}
          </Card>
        </Col>
      </Row>

      <Card title="Recent Experiments" className="recent-experiments">
        {experiments.length > 0 ? (
          <Table 
            columns={columns} 
            dataSource={experiments} 
            rowKey="id"
            pagination={{ pageSize: 5 }}
          />
        ) : (
          <Empty 
            description="No experiments yet" 
            image={Empty.PRESENTED_IMAGE_SIMPLE} 
          >
            <Button type="primary">
              <Link to="/new-experiment">Create your first experiment</Link>
            </Button>
          </Empty>
        )}
      </Card>
    </div>
  );
}

export default Dashboard;