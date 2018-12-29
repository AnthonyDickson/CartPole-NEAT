import React, {Component} from 'react';
import {Button, ListGroup, ListGroupItem} from "reactstrap";
import Moment from 'react-moment';
import 'moment-timezone';

const API = "http://localhost:5000/api";
const TIMESTAMP_FORMAT = "YYYY-MM-DD hh:mm A";

class RunList extends Component {
    render() {
        if (this.props.data.length < 1) {
            return <div>No data.</div>
        }

        return (
            <ListGroup>
                {
                    this.props.data.map((item, i) =>
                        <ListGroupItem key={i}>
                            <RunListItem i={i} item={item} handleDelete={this.props.handleDelete}/>
                        </ListGroupItem>
                    )
                }
            </ListGroup>
        );
    }
}

class RunListItem extends Component {
    render() {
        const start_date = <Moment date={this.props.item.start_date} format={TIMESTAMP_FORMAT}/>;
        const end_date = (this.props.item.end_date === null) ? <span>-</span> :
            <Moment date={this.props.item.end_date} format={TIMESTAMP_FORMAT}/>;

        return (
            <ul className="list-inline">
                <li className="list-inline-item">{this.props.item.id}</li>
                <li className="list-inline-item">{start_date}</li>
                <li className="list-inline-item">{end_date}</li>
                <li className="list-inline-item float-right">
                    <Button onClick={() => this.props.handleDelete(this.props.item.id)} color="danger">Delete</Button>
                </li>
            </ul>
        );
    }
}

class Dashboard extends Component {
    constructor(props) {
        super(props);

        this.state = {
            isLoading: true,
            data: null
        };

        this.delete = this.delete.bind(this);
        this.update = this.update.bind(this);
        setInterval(this.update, 60 * 1000);
    }

    componentDidMount() {
        this.update();
    }

    update() {
        fetch(API + "/runs")
            .then(response => response.json())
            .then(response => this.setState({
                isLoading: false,
                data: response.data
            }));
    }

    delete(id) {
        fetch(API + '/runs/' + id, {method: "DELETE"})
            .then(response => {
                if (response.status === 204) {
                    this.setState({data: this.state.data.filter(elem => elem.id !== id)});
                }
            });
    }

    render() {
        if (this.state.isLoading) {
            return <div>Loading...</div>
        }

        return (
            <>
                <Button onClick={this.update}>Update</Button>
                <RunList data={this.state.data} handleDelete={this.delete}/>
            </>
        );
    }
}

export {Dashboard};
