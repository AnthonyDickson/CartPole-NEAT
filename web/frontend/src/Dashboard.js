import React, {Component} from 'react';
import {ListGroup, ListGroupItem} from "reactstrap";
import Moment from 'react-moment';
import 'moment-timezone';

const API = "http://localhost:5000/api";

class RunList extends Component {
    render() {
        if (this.props.data.length < 1) {
            return <div>No data.</div>
        }

        const end_date = (this.props.data.end_date == null) ?
            <span>-</span> :
            <Moment date={this.props.data.end_date} format="YYYY-MM-DD hh:mm A"/>;

        return (
            <ListGroup>
                {
                    this.props.data.map((item, i) => {
                        return <ListGroupItem key={i}>
                            <span>{item.id} </span>
                            <Moment date={item.start_date} format="YYYY-MM-DD hh:mm A "/>
                            {end_date}
                        </ListGroupItem>
                    })
                }
            </ListGroup>
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
    }

    componentDidMount() {
        fetch(API + "/runs")
            .then(response => response.json())
            .then(response => this.setState({
                isLoading: false,
                data: response.data
            }))
    }

    render() {
        if (this.state.isLoading) {
            return <div>Loading...</div>
        }

        return (
            <RunList data={this.state.data}/>
        );
    }
}

export {Dashboard};
